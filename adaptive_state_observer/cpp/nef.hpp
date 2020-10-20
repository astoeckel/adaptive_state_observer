/*
 *  C++ NEF Implementation
 *  Copyright (C) 2019-2020  Andreas Stöckel
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
#include <random>
#include <iostream>

#include "dists.hpp"
#include "matrix.hpp"
#include "nef.hpp"

/**
 * Structure defining the ReLU nonlinearity.
 */
struct ReLU {
	static double inverse(double a) { return a; }

	static double activity(double x) { return std::max(x, 0.0); }

	static double derivative(double x, double) { return (x > 0.0) ? 1.0 : 0.0; }
};

/**
 * Structure defining the LIF nonlinearity.
 */
template <int SLOPE_NUM = 2, int SLOPE_DEN = 3>
struct LIF {
	static constexpr double slope = double(SLOPE_NUM) / double(SLOPE_DEN);

	static double inverse(double a)
	{
		if (a > 0.0) {
			return 1.0 / (1.0 - std::exp(slope - (1.0 / a)));
		}
		return 0.0;
	}

	static double activity(double x)
	{
		if (x >= 1.0 + 1e-6) {
			return 1.0 / (slope - std::log1p(-1.0 / x));
		}
		return 0.0;
	}

	static double derivative(double x, double a)
	{
		if (x >= 1.0 + 1e-6) {
			return (a * a) / (x * x * (1.0 - 1.0 / x));
		}
		return 0.0;
	}
};

/**
 * Two-layer neural network with a fixed input layer and a learned output layer.
 */
template <typename NEURON_TYPE, size_t N_NEURONS, size_t N_DIM_IN,
          size_t N_DIM_OUT>
class NEFEnsemble {
public:
	static constexpr size_t n_neurons = N_NEURONS;
	static constexpr size_t n_dim_in = N_DIM_IN;
	static constexpr size_t n_dim_out = N_DIM_OUT;
	static constexpr size_t n_params = N_DIM_IN + 2;

	using Parameters = Mat<n_neurons, n_params>;
	using NeuronVec = Arr<n_neurons>;
	using Input = Vec<n_dim_in>;
	using Output = Vec<n_dim_out>;
	using Weights = Mat<n_dim_out, n_neurons>;
	using Jacobian = Mat<n_dim_out, n_dim_in>;
	using Encoder = Mat<n_neurons, n_dim_in>;

private:
	Parameters m_params;
	Weights m_weights;

	Eigen::Map<NeuronVec, Eigen::Aligned16, Stride<NeuronVec, n_params>> m_gain;
	Eigen::Map<NeuronVec, Eigen::Aligned8, Stride<NeuronVec, n_params>> m_bias;
	Eigen::Map<Encoder, Eigen::Aligned16, Stride<Encoder, n_params>> m_encoder;

public:
	NEFEnsemble(std::mt19937 &rng)
	    : m_params(Parameters::Zero()),
	      m_weights(Weights::Zero()),
	      m_gain(m_params.data() + 0),
	      m_bias(m_params.data() + 1),
	      m_encoder(m_params.data() + 2)
	{
		// Compute the intercepts and maximum rates
		const NeuronVec intercepts =
		    Dists::uniform(-0.9, 0.9, n_neurons, 1, rng);
		const NeuronVec max_rates = Dists::uniform(0.5, 1.0, n_neurons, 1, rng);

		// Compute the currents corresponding to the intercept and the maximum
		// rates
		const double j_0 = NEURON_TYPE::inverse(1e-6);
		NeuronVec j_max_rates;
		for (size_t i = 0; i < n_neurons; i++) {
			j_max_rates[i] = NEURON_TYPE::inverse(max_rates[i]);
		}

		// Use the currents to compute the gain and bias
		m_gain = (j_0 - j_max_rates) / (intercepts - 1.0);
		m_bias = j_max_rates - m_gain;
		m_encoder = Dists::hypersphere(n_neurons, n_dim_in, rng);
	}

	/**
	 * Computes the input current flowing into each non-linearity.
	 */
	NeuronVec activations(const Input &x) const
	{
		return m_gain * (m_encoder * x).array() + m_bias;
	}

	/**
	 * Computes the activity of each neuron after the non-linearity.
	 */
	NeuronVec activities(const NeuronVec &J) const
	{
		NeuronVec res;
		for (size_t i = 0; i < n_neurons; i++) {
			res[i] = NEURON_TYPE::activity(J[i]);
		}
		return res;
	}

	NeuronVec derivative(const NeuronVec &J, const NeuronVec &a) const
	{
		NeuronVec res;
		for (size_t i = 0; i < n_neurons; i++) {
			res[i] = m_gain[i] * NEURON_TYPE::derivative(J[i], a[i]);
		}
		return res;
	}

	Output forward(const NeuronVec &a) const { return m_weights * a.matrix(); }

	void backward(const NeuronVec &a, const Output &err, double eta)
	{
		m_weights += (eta * err) * a.matrix().transpose();
	}

	Jacobian jacobian(const NeuronVec &J, const NeuronVec &a,
	                  const Input &) const
	{
		Jacobian res;

		// The Jacobian is given as
		//  ∂
		// ---- W @ a(E @ x) = W @ diag(da(E @ x)) @ E = W @ (da(E @ x) * E)
		//  ∂x

		// Evaluate the derivative of the activities given the context
		const NeuronVec da = derivative(J, a);

		// Compute each column of the target matrix individually. This way we
		// don't have to allocate a large intermediate matrix on the heap
		for (size_t i = 0; i < n_dim_in; i++) {
			// Multiply the derivative of the activities by the corresponding
			// column of the encoder.
			const NeuronVec tmp = da * m_encoder.col(i).array();

			// Compute the matrix-vector product between the temporary scaled
			// column of the encoding matrix and multiply it by the decoding
			// weights
			res.col(i).noalias() = m_weights * tmp.matrix();
		}

		return res;
	}

	Parameters &params() { return m_params; }
	const Parameters &params() const { return m_params; }

	Weights &weights() { return m_weights; }
	const Weights &weights() const { return m_weights; }
};

template <size_t N_NEURONS, size_t N_DIM_IN, size_t N_DIM_OUT>
using ReLUEnsemble = NEFEnsemble<ReLU, N_NEURONS, N_DIM_IN, N_DIM_OUT>;

template <size_t N_NEURONS, size_t N_DIM_IN, size_t N_DIM_OUT>
using LIFEnsemble = NEFEnsemble<LIF<>, N_NEURONS, N_DIM_IN, N_DIM_OUT>;

template <size_t N_NEURONS, size_t N_DIM_IN, size_t N_DIM_OUT>
class RBFEnsemble {
public:
	static constexpr size_t n_neurons = N_NEURONS;
	static constexpr size_t n_dim_in = N_DIM_IN;
	static constexpr size_t n_dim_out = N_DIM_OUT;
	static constexpr size_t n_params = N_DIM_IN + 1;

	using Parameters = Mat<n_neurons, n_params>;
	using NeuronVec = Arr<n_neurons>;
	using Input = Vec<n_dim_in>;
	using Output = Vec<n_dim_out>;
	using Weights = Mat<n_dim_out, n_neurons>;
	using Jacobian = Mat<n_dim_out, n_dim_in>;
	using Centroids = Mat<n_neurons, n_dim_in>;

private:
	Parameters m_params;
	Weights m_weights;

	Eigen::Map<NeuronVec, Eigen::Aligned8, Stride<NeuronVec, n_params>>
	    m_sigma_sq;
	Eigen::Map<Centroids, Eigen::Aligned16, Stride<Centroids, n_params>> m_mu;

public:
	RBFEnsemble(std::mt19937 &rng)
	    : m_params(Parameters::Zero()),
	      m_weights(Weights::Zero()),
	      m_sigma_sq(m_params.data() + n_dim_in),
	      m_mu(m_params.data() + 0)
	{
		m_sigma_sq = Eigen::pow(
		    10.0, 2.0 * Dists::uniform(-1.0, 0.0, n_neurons, 1, rng).array());
		m_mu = Dists::halton(n_neurons, n_dim_in).array() * 2.0 - 1.0;
	}

	/**
	 * Computes the input current flowing into each non-linearity.
	 */
	NeuronVec activations(const Input &x) const
	{
		NeuronVec res;
		for (size_t i = 0; i < n_neurons; i++) {
			res[i] = -(m_mu.row(i) - x.transpose()).squaredNorm() / m_sigma_sq[i];
		}
		return res;
	}

	/**
	 * Computes the input current flowing into each non-linearity.
	 */
	NeuronVec activities(const NeuronVec &a) const { return a.exp(); }

	Output forward(const NeuronVec &a) const { return m_weights * a.matrix(); }

	void backward(const NeuronVec &a, const Output &err, double eta)
	{
		m_weights += (eta * err) * a.matrix().transpose();
	}

	Jacobian jacobian(const NeuronVec &, const NeuronVec &a,
	                  const Input &x) const
	{
		Jacobian res;

		// Evaluate the derivative of the activities given the context
		const NeuronVec da = 2.0 * a / m_sigma_sq;

		// Compute each column of the target matrix individually. This way we
		// don't have to allocate a large intermediate matrix on the heap
		for (size_t i = 0; i < n_dim_in; i++) {
			// Multiply the derivative of the activities by the corresponding
			// column of the encoder.
			const NeuronVec tmp = da * (m_mu.col(i).array() - x[i]);

			// Compute the matrix-vector product between the temporary scaled
			// column of the encoding matrix and multiply it by the decoding
			// weights
			res.col(i).noalias() = m_weights * tmp.matrix();
		}

		return res;
	}

	Parameters &params() { return m_params; }
	const Parameters &params() const { return m_params; }

	Weights &weights() { return m_weights; }
	const Weights &weights() const { return m_weights; }
};

