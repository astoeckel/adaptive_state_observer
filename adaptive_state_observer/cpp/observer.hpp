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

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>

#include "matrix.hpp"
#include "nef.hpp"

template <size_t N_STATE_DIM, size_t N_AUX_STATE_DIM, size_t N_OBSERVATION_DIM,
          size_t N_CONTROL_DIM, size_t N_NEURONS_G, size_t N_NEURONS_H,
          template <size_t, size_t, size_t> class ENSEMBLE_TYPE = ReLUEnsemble>
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
	using EnsembleG = ENSEMBLE_TYPE<N_NEURONS_G, N + U, Nr>;
	using EnsembleH = ENSEMBLE_TYPE<N_NEURONS_H, Nr, M>;

	/* Neuron ensemble specific vector types */
	using NeuronVecG = typename EnsembleG::NeuronVec;
	using NeuronVecH = typename EnsembleH::NeuronVec;

private:
	StateTransitionMatrix m_G0;
	ObservationMatrix m_H0;

	EnsembleG m_ens_g;
	EnsembleH m_ens_h;

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
	      m_ens_h(rng)
	{
	}

	State pred_dx(const State &x, const Control &u) const
	{
		// Concatenate the current state and the control signal
		const StateAndControl xu = concatenate_state_and_control(x, u);

		// Create the output arrays; dxr corresponds to a part of dx
		State dx = State::Zero();
		Eigen::Map<ReducedState> dxr(dx.data());

		// Evaluate the linear part
		dx = m_G0 * xu;

		// Evaluate the neural network
		const NeuronVecG J_g = m_ens_g.activations(xu);
		const NeuronVecG a_g = m_ens_g.activities(J_g);
		dxr += m_ens_g.forward(a_g);

		return dx;
	}

	Observation pred_z(const State &x) const
	{
		// We're only looking at the first few dimensions of the state vector
		Eigen::Map<const ReducedState> xr(x.data());

		// Compute the predicted observation
		const NeuronVecH J_h = m_ens_h.activations(xr);
		const NeuronVecH a_h = m_ens_h.activities(J_h);
		return m_H0 * xr + m_ens_h.forward(a_h);
	}

	/**
	 * Takes the last state x, the control input u, and the observation z to
	 * compute a new state x'. Updates the learned functions.
	 */
	std::tuple<State, Observation> step(const State &x, const Control &u,
	                                    const Observation &z, const Params &p)
	{
		// Build several variants of the state vector
		const StateAndControl xu = concatenate_state_and_control(x, u);
		Eigen::Map<const ReducedState> xr(x.data());

		// Compute the neuron ensemble activations (input currents)
		const NeuronVecG J_g = m_ens_g.activations(xu);
		const NeuronVecH J_h = m_ens_h.activations(xr);

		// Compute the neuron ensemble activities
		const NeuronVecG a_g = m_ens_g.activities(J_g);
		const NeuronVecH a_h = m_ens_h.activities(J_h);

		// Compute the observation error, and derive the state update error
		Observation pred_z = m_H0 * xr + m_ens_h.forward(a_h);
		const ObservationMatrix H_ = m_H0 + m_ens_h.jacobian(J_h, a_h, xr);
		const Observation err_z = z - pred_z;
		const ReducedState err_dxr = p.gain * (H_.transpose() * err_z);

		// Compute the predicted state update; provide a view on the first
		// "reduced" dimensions of the state
		State pred_dx = m_G0 * xu;
		Eigen::Map<ReducedState> pred_dxr(pred_dx.data());
		pred_dxr += m_ens_g.forward(a_g);

		// Update the learned functions according to the computed errors
		m_ens_g.backward(a_g, err_dxr, p.dt * p.eta * p.eta_rel_g);
		m_ens_h.backward(a_h, err_z, p.dt * p.eta * p.eta_rel_h);

		// Compute the complete state update by summing the predicted state
		// update and the state update error.
		State dx = pred_dx;
		Eigen::Map<ReducedState> dxr(dx.data());
		dxr += err_dxr;

		// Compute the actual new state
		return std::make_tuple<State, Observation>(x + p.dt * dx,
		                                           std::move(pred_z));
	}

	double *G0() { return m_G0.data(); }
	double *H0() { return m_H0.data(); }

	EnsembleG &ens_g() { return m_ens_g; }
	EnsembleH &ens_h() { return m_ens_h; }
};

