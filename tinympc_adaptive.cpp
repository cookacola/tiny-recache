#include "tinympc_adaptive.hpp"
#include <cmath>

// Use the inline constants from the namespace
using namespace QuadParams;

// ----------------------------------------------------
// Helpers
// ----------------------------------------------------
Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
    // Python: hat(v) = [[0, -v2, v1],
    //                   [v2, 0,  -v0],
    //                   [-v1, v0, 0]]
    Eigen::Matrix3d V;
    V <<   0.0,   -v.z(),  v.y(),
           v.z(),  0.0,    -v.x(),
          -v.y(),  v.x(),   0.0;
    return V;
}

// L(q): 4x4
Eigen::Matrix<double, 4, 4> L(const Eigen::Vector4d& q) {
    double s = q(0);
    Eigen::Vector3d v = q.segment<3>(1);

    // up = [s, -v.x, -v.y, -v.z]
    // down = [ v ; sI + hat(v) ] => 3x4
    // final L is 4x4
    Eigen::Matrix<double, 4, 4> Lmat;
    Lmat.setZero();

    // First row
    Lmat(0, 0) = s;
    Lmat(0, 1) = -v.x();
    Lmat(0, 2) = -v.y();
    Lmat(0, 3) = -v.z();

    // Next 3 rows
    // top-left 3x1 is v
    Lmat.block<3,1>(1,0) = v;

    // s*I + hat(v)
    Eigen::Matrix3d sI_hatv = s * Eigen::Matrix3d::Identity() + hat(v);
    Lmat.block<3,3>(1,1) = sI_hatv;

    return Lmat;
}

// T = diag(1, -1, -1, -1)
static const Eigen::Matrix4d Tmat = [](){
    Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
    temp(1,1) = -1.0;
    temp(2,2) = -1.0;
    temp(3,3) = -1.0;
    return temp;
}();

const Eigen::Matrix4d& T() {
    return Tmat;
}

// H = [ [0,0,0],
//       [1,0,0],
//       [0,1,0],
//       [0,0,1] ]
static const Eigen::Matrix<double,4,3> Hmat = [](){
    Eigen::Matrix<double,4,3> temp;
    temp.setZero();
    temp.block<3,3>(1,0) = Eigen::Matrix3d::Identity();
    return temp;
}();

const Eigen::Matrix<double,4,3>& H() {
    return Hmat;
}

// qtoQ(q) = H^T * T * L(q) * T * L(q) * H
Eigen::Matrix3d qtoQ(const Eigen::Vector4d& q) {
    Eigen::Matrix<double, 4, 4> Lq = L(q);
    // T * L(q)
    Eigen::Matrix<double,4,4> TLq = T() * Lq;
    // T * L(q) * T
    Eigen::Matrix<double,4,4> TLqT = TLq * T();
    // T * L(q) * T * L(q)
    Eigen::Matrix<double,4,4> TLqT_Lq = TLqT * Lq;
    // Finally multiply with H on left and H^T on right
    // Actually in code: H^T * T * L(q) * T * L(q) * H is 3x3
    // Check carefully the order in Python: H^T @ T @ L(q) @ T @ L(q) @ H
    Eigen::Matrix3d Qres = H().transpose() * (TLqT_Lq * H());
    return Qres;
}

// G(q) = L(q)*H => 4x3
Eigen::Matrix<double, 4, 3> G(const Eigen::Vector4d& q) {
    return L(q) * H();
}

// rp to q
Eigen::Vector4d rptoq(const Eigen::Vector3d& phi) {
    double denom = std::sqrt(1.0 + phi.dot(phi));
    Eigen::Vector4d q;
    q(0) = 1.0 / denom;
    q.segment<3>(1) = phi / denom;
    return q;
}

// q to rp
Eigen::Vector3d qtorp(const Eigen::Vector4d& q) {
    // q(0) is scalar, q(1..3) is vector
    double s = q(0);
    Eigen::Vector3d v = q.segment<3>(1);
    return (1.0 / s) * v;
}

// E(q): 13x12
Eigen::Matrix<double, 13, 12> E(const Eigen::Vector4d& q) {
    // E = [ I3, 0,    0    ]  => 3x12
    //       [ 0, G(q), 0 ]  => 4x12
    //       [ 0, 0,    I6 ] => 6x12
    //
    // We'll construct row-by-row.

    Eigen::Matrix<double, 13, 12> Emat;
    Emat.setZero();

    // top: I3 (3x3), then 9 zeros. So (3,3) block is Identity
    Emat.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

    // middle: 4x3 zeros, then G(q) which is 4x3, then 4x6 zeros
    // G(q) is 4x3, place it at (3,3)
    Eigen::Matrix<double, 4, 3> Gq = G(q);
    Emat.block<4,3>(3,3) = Gq;

    // bottom: 6x6 identity placed at (7,6)
    Emat.block<6,6>(7,6) = Eigen::Matrix<double,6,6>::Identity();

    return Emat;
}

// ----------------------------------------------------
// quad_dynamics and RK4
// ----------------------------------------------------
Eigen::Matrix<double, 13, 1> quad_dynamics(const Eigen::Matrix<double, 13, 1>& x,
                                           const Eigen::Vector4d& u,
                                           const Eigen::Vector3d& wind)
{
    // x = [r(0..2), q(3..6), v(7..9), omg(10..12)]
    // Make sure q is normalized if needed
    // But we'll do it in the calling code or assume near-normalized.

    Eigen::Vector3d r = x.segment<3>(0);
    Eigen::Vector4d q = x.segment<4>(3); 
    // Normalize just in case
    q.normalize();
    Eigen::Vector3d v = x.segment<3>(7);
    Eigen::Vector3d omg = x.segment<3>(10);

    // Q(q) is 3x3
    Eigen::Matrix3d Qq = qtoQ(q);

    // dr = v
    Eigen::Vector3d dr = v;

    // dq = 0.5 * L(q)*H * omg
    // L(q) is 4x4, H is 4x3, so L(q)*H is 4x3, multiply by omg(3x1) => 4x1
    Eigen::Matrix<double,4,3> LH = L(q) * H();
    Eigen::Vector4d dq = 0.5 * LH * omg;

    // dv = [0, 0, -g] + (1/mass)*Q(q)*[[0,0,0,0],[0,0,0,0],[kt,kt,kt,kt]]*u + wind
    Eigen::Vector3d gravity(0.0, 0.0, -g);
    // Construct 3x4 thrust matrix Tmat for the force
    Eigen::Matrix<double,3,4> thrustMat;
    thrustMat.setZero();
    thrustMat(2,0) = kt;
    thrustMat(2,1) = kt;
    thrustMat(2,2) = kt;
    thrustMat(2,3) = kt;

    Eigen::Vector3d dv = gravity + (1.0 / mass) * (Qq * (thrustMat * u)) + wind;

    // domega = inv(J)*(-hat(omg)*J*omg + [torqueMatrix]*u)
    // torque matrix is [ [-el*kt, -el*kt, +el*kt, +el*kt],
    //                    [-el*kt,  el*kt,  el*kt, -el*kt],
    //                    [   -km,    km,    -km,    km ] ]
    Eigen::Matrix<double, 3, 4> torqueMat;
    torqueMat << 
         -el*kt,  -el*kt,   el*kt,   el*kt,
         -el*kt,   el*kt,   el*kt,  -el*kt,
            -km,      km,     -km,      km;

    Eigen::Matrix3d Jinv = J.inverse();
    Eigen::Vector3d domega = Jinv * ( -hat(omg)*J*omg + torqueMat*u );

    // Return 13x1
    Eigen::Matrix<double, 13, 1> xdot;
    xdot << dr, dq, dv, domega;
    return xdot;
}

// 4th-order Runge-Kutta
Eigen::Matrix<double, 13, 1> quad_dynamics_rk4(const Eigen::Matrix<double, 13, 1>& x,
                                               const Eigen::Vector4d& u)
{
    // step = h
    Eigen::Matrix<double, 13, 1> f1 = quad_dynamics(x, u);
    Eigen::Matrix<double, 13, 1> f2 = quad_dynamics(x + 0.5 * h * f1, u);
    Eigen::Matrix<double, 13, 1> f3 = quad_dynamics(x + 0.5 * h * f2, u);
    Eigen::Matrix<double, 13, 1> f4 = quad_dynamics(x + h * f3, u);

    Eigen::Matrix<double, 13, 1> xn = x + (h/6.0)*(f1 + 2.0*f2 + 2.0*f3 + f4);

    // Re-normalize quaternion part
    Eigen::Vector4d qn = xn.segment<4>(3);
    qn.normalize();
    xn.segment<4>(3) = qn;

    return xn;
}

