#pragma once

#include <Eigen/Dense>

/**
 *  \file tinympc_adaptive.hpp
 *  \brief Core math functions from tinyMPC hover adaptive example, ported to C++.
 *
 *  Requires Eigen (http://eigen.tuxfamily.org/).
 */

// ----------------------------------------------------
// Constants & Parameters
// ----------------------------------------------------
namespace QuadParams {
    inline constexpr double mass = 0.035;         // quadrotor mass
    inline constexpr double g    = 9.81;          // gravity
    inline constexpr double thrustToTorque = 0.0008;
    inline constexpr double scale = 65535.0;
    inline constexpr double kt = 2.245365e-6 * scale;
    inline constexpr double km = kt * thrustToTorque;

    // Distance from center to motor (for torque matrix).
    // 0.046 / sqrt(2) from the Python code:
    inline constexpr double el = 0.046 / 1.414213562;

    // Inertia matrix
    // J is 3x3
    inline const Eigen::Matrix3d J = (Eigen::Matrix3d() <<
        1.66e-5, 0.83e-6, 0.72e-6,
        0.83e-6, 1.66e-5, 1.8e-6,
        0.72e-6, 1.8e-6, 2.93e-5).finished();

    // Integration step (freq = 50 => h = 0.02)
    inline constexpr double freq = 50.0;
    inline constexpr double h = 1.0 / freq;
}

// ----------------------------------------------------
// Function Declarations
// ----------------------------------------------------
Eigen::Matrix3d hat(const Eigen::Vector3d& v);

/**
 * L(q) operator for quaternions.
 *  q = (s, v), s = q[0], v = q[1:4]
 */
Eigen::Matrix<double, 4, 4> L(const Eigen::Vector4d& q);

/**
 * The diagonal matrix T = diag(1.0, -1.0, -1.0, -1.0)
 *  We'll return it as a static constant.
 */
const Eigen::Matrix4d& T();

/**
 * The matrix H = [0_{1x3}; I_{3x3}] in 4x3 form.
 *  We'll return it as a static constant.
 */
const Eigen::Matrix<double, 4, 3>& H();

/**
 * Compute Q(q) = H^T * T * L(q) * T * L(q) * H (3x3).
 */
Eigen::Matrix3d qtoQ(const Eigen::Vector4d& q);

/**
 * Compute G(q) = L(q)*H. This is 4x3.
 */
Eigen::Matrix<double, 4, 3> G(const Eigen::Vector4d& q);

/**
 * Convert an rpy-like "phi" in R^3 into a unit quaternion, i.e. rp to q.
 *  rptoq(phi) = (1 / sqrt(1 + phi^T phi)) * [1, phi].
 */
Eigen::Vector4d rptoq(const Eigen::Vector3d& phi);

/**
 * Convert a quaternion q = (s, v) with s = q[0], v = q[1:4] to rp (vector part / scalar part).
 *  qtorp(q) = v / s
 */
Eigen::Vector3d qtorp(const Eigen::Vector4d& q);

/**
 * Compute E(q), the linearization transform from 13 states to 12 (with quaternion).
 *  E(q) is 13x12, but we often transpose it.  For 4D quaternion, 3D velocity, 3D omega, etc.
 * 
 *  E = [ I_{3x3}, 0, 0;
 *        0,      G(q), 0;
 *        0,      0,    I_{6x6} ]
 * 
 *  This returns a 13x12. 
 */
Eigen::Matrix<double, 13, 12> E(const Eigen::Vector4d& q);

/**
 * quad_dynamics
 * 
 * x: 13D state [r(0:3), q(3:7), v(7:10), omega(10:13)]
 * u: 4D input  [motor commands]
 * wind: 3D wind disturbance
 * 
 * returns dx/dt as 13D
 */
Eigen::Matrix<double, 13, 1> quad_dynamics(const Eigen::Matrix<double, 13, 1>& x,
                                           const Eigen::Vector4d& u,
                                           const Eigen::Vector3d& wind = Eigen::Vector3d::Zero());

/**
 * quad_dynamics_rk4
 *  One step of RK4 integration using h = 1/freq (0.02s).
 */
Eigen::Matrix<double, 13, 1> quad_dynamics_rk4(const Eigen::Matrix<double, 13, 1>& x,
                                               const Eigen::Vector4d& u);

