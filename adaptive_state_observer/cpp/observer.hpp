/*
 *  C++ Adaptive State Observer
 *  Copyright (C) 2020  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <iostream>

#include <cstdint>
#include <cstddef>
#include <random>
#include <tuple>

#include <Eigen/Dense>

#include "nef.hpp"

template <size_t N_STATE_DIM, size_t N_AUX_STATE_DIM, size_t N_OBSERVATION_DIM,
          size_t N_CONTROL_DIM, size_t N_NEURONS_G, size_t N_NEURONS_H,
          typename NEURON_TYPE = ReLU>
class AdaptiveStateObserver {
public:
	/**
	 * Set of parameters controlling the adaptive state observer at runtime.
	 */
	struct Params {
		double dt;
		double gain;
		double eta;
		double eta_rel_g;
		double eta_rel_h;

		Params()
		    : dt(1e-3), gain(10.0), eta(1e-2), eta_rel_g(0.1), eta_rel_h(1.0)
		{
		}
	};

	static constexpr size_t N = N_STATE_DIM;
	static constexpr size_t Nr = N_STATE_DIM - N_AUX_STATE_DIM;
	static constexpr size_t M = N_OBSERVATION_DIM;
	static constexpr size_t U = N_CONTROL_DIM;

	template <size_t M, size_t N>
	using Mat = Eigen::Matrix<double, M, N, Eigen::RowMajor>;

	template <size_t M>
	using Vec = Eigen::Matrix<double, M, 1>;

	/* Vector types */
	using State = Vec<N>;
	using ReducedState = Vec<Nr>;
	using Observation = Vec<M>;
	using Control = Vec<U>;
	using StateAndControl = Vec<N + U>;

	/* Matrix types */
	using StateTransitionMatrix = Mat<N, N + U>;
	using ObservationMatrix = Mat<M, Nr>;

	/* Neuron ensemble types */
	using EnsembleG = Ensemble<NEURON_TYPE, N_NEURONS_G, N + U>;
	using EnsembleH = Ensemble<NEURON_TYPE, N_NEURONS_H, Nr>;

	/* Neuron ensemble specific vector types */
	using ActivitiesG = typename EnsembleG::Activities;
	using ActivitiesH = typename EnsembleH::Activities;

	/* Learning rule implementations */
	using LearnG = PES<N_NEURONS_G, Nr>;
	using LearnH = PES<N_NEURONS_H, M>;

private:
	StateTransitionMatrix m_G0;
	ObservationMatrix m_H0;

	EnsembleG m_ens_g;
	EnsembleH m_ens_h;

	LearnG m_learn_g;
	LearnH m_learn_h;

	/**
	 * Helper function used internally to generate a vector combining the given
	 * state and control signal.
	 */
	static StateAndControl concatenate_state_and_control(const State &x,
	                                                     const Control &u)
	{
		// Local vector containing both the state and the control as one vector
		StateAndControl xu;

		// Copy the given data into the common vector
		for (size_t i = 0; i < N; i++) {
			xu[i] = x[i];
		}
		for (size_t i = 0; i < U; i++) {
			xu[i + N] = u[i];
		}

		return xu;
	}

public:
	AdaptiveStateObserver(std::mt19937 &rng)
	    : m_G0(StateTransitionMatrix::Zero()),
	      m_H0(ObservationMatrix::Zero()),
	      m_ens_g(rng),
	      m_ens_h(rng),
	      m_learn_g(1.0),
	      m_learn_h(1.0)
	{
	}

	/**
	 * Evaluates the learned dynamics g(x, u) for the given (full) state vector
	 * x and the control input u.
	 */
	ReducedState g(const State &x, const Control &u) const
	{
		// Compute the activities of ens_g and decode according to the learned
		// function weights; add the dynamics described by G0
		const StateAndControl xu = concatenate_state_and_control(x, u);
		return m_learn_g.weights() * m_ens_g(xu);
	}

	/**
	 * Computes the learned observation prediction h(xr) for the given (reduced)
	 * state vector xr.
	 */
	Observation h(const ReducedState &xr) const
	{
		return m_learn_h.weights() * m_ens_h(xr);
	}

	/**
	 * Computes the Jacobian of the learned observation function at the given
	 * point xr.
	 */
	ObservationMatrix H(const ReducedState &xr) const
	{
		ObservationMatrix H;

		// The Jacobian is given as
		//  ∂
		// ---- W @ a(E @ x) = W @ diag(da(E @ x)) @ E = W @ (da(E @ x) * E)
		//  ∂x

		// Evaluate the derivative of the activities given the context
		const ActivitiesH da = m_ens_h.derivative(xr);

		// Compute each column of the target matrix individually. This way we
		// don't have to allocate a large intermediate matrix on the heap
		for (size_t i = 0; i < Nr; i++) {
			// Multiply the derivative of the activities by the corresponding
			// column of the encoder.
			const ActivitiesH t = da.array() * m_ens_h.encoder().col(i).array();

			// Compute the matrix-vector product between the temporary scaled
			// column of the encoding matrix and multiply it by the decoding
			// weights
			H.col(i).noalias() = m_learn_h.weights() * t;
		}

		return H;
	}

	State pred_dx(const State &x, const Control &u) const {
		// Concatenate the current state and the control signal
		const StateAndControl xu = concatenate_state_and_control(x, u);

		// Create the output arrays; dxr corresponds to a part of dx
		State dx = State::Zero();
		Eigen::Map<ReducedState> dxr(dx.data());

		// Evaluate the linear part
		dx = m_G0 * xu;

		// Evaluate the neural network
		dxr += m_learn_g.weights() * m_ens_g(xu);

		return dx;
	}

	Observation pred_z(const State &x) const {
		// We're only looking at the first few dimensions of the state vector
		Eigen::Map<const ReducedState> xr(x.data());

		// Compute the predicted observation
		return m_H0 * xr + m_learn_h.weights() * m_ens_h(xr);
	}

	/**
	 * Takes the last state x, the control input u, and the observation z to
	 * compute a new state x'. Updates the learned functions.
	 */
	std::tuple<State, Observation>
	step(const State &x, const Control &u, const Observation &z, const Params &p)
	{
		// Build several variants of the state vector
		const StateAndControl xu = concatenate_state_and_control(x, u);
		Eigen::Map<const ReducedState> xr(x.data());

		// Compute the neuron ensemble activities
		const ActivitiesG a_g = m_ens_g(xu);
		const ActivitiesH a_h = m_ens_h(xr);

		// Compute the observation error, and derive the state update error
		Observation pred_z = m_H0 * xr + m_learn_h.weights() * a_h;
		const ObservationMatrix H_ = m_H0 + H(xr);
		const Observation err_z = z - pred_z;
		const ReducedState err_dxr = p.gain * (H_.transpose() * err_z);

		// Compute the predicted state update; provide a view on the first
		// "reduced" dimensions of the state
		State pred_dx = m_G0 * xu;
		Eigen::Map<ReducedState> pred_dxr(pred_dx.data());
		pred_dxr += m_learn_g.weights() * a_g;

		// Update the learned functions according to the computed errors
		m_learn_g.step(a_g, err_dxr, p.dt * p.eta * p.eta_rel_g);
		m_learn_h.step(a_h, err_z, p.dt * p.eta * p.eta_rel_h);

		// Compute the complete state update by summing the predicted state
		// update and the state update error.
		State dx = pred_dx;
		Eigen::Map<ReducedState> dxr(dx.data());
		dxr += err_dxr;

		// Compute the actual new state
		return std::make_tuple<State, Observation>(x + p.dt * dx, std::move(pred_z));
	}

	double *G0() { return m_G0.data(); }
	double *H0() { return m_H0.data(); }

	double *g_W() { return m_learn_g.weights().data(); }
	double *h_W() { return m_learn_h.weights().data(); }
	double *g_E() { return m_ens_g.encoder().data(); }
	double *h_E() { return m_ens_h.encoder().data(); }
	double *g_gain() { return m_ens_g.gain().data(); }
	double *h_gain() { return m_ens_h.gain().data(); }
	double *g_bias() { return m_ens_g.bias().data(); }
	double *h_bias() { return m_ens_h.bias().data(); }
};

