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

#include <cstddef>
#include <random>

#include <Eigen/Dense>

#include "dists.hpp"

/**
 * The PES class implements the prescribed error sensitivity learning rule.
 *
 * @tparam N_NEURONS is the number of pre-neurons.
 * @tparam N_DIM is the number of dimensions that should be decoded from the
 *         pre-neurons.
 */
template <size_t N_NEURONS, size_t N_DIM>
class PES {
public:
	static constexpr size_t n_neurons = N_NEURONS;
	static constexpr size_t n_dim = N_DIM;

	using Weights =
	    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using Decoded = Eigen::Matrix<double, N_DIM, 1>;
	using Error = Eigen::Matrix<double, N_DIM, 1>;
	using Activities = Eigen::Matrix<double, N_NEURONS, 1>;

private:
	Weights m_weights;
	double m_learning_rate;

public:
	PES(double learning_rate = 1e-2)
	    : m_weights(Weights::Zero(n_dim, N_NEURONS)),
	      m_learning_rate(learning_rate)
	{
	}

	void step(const Activities &a, const Error &e, double dt = 1e-3)
	{
		m_weights += dt * m_learning_rate * e * a.transpose();
	}

	double learning_rate() const { return m_learning_rate; }

	Decoded evaluate(const Activities &a) const { return m_weights * a; }

	Decoded operator()(const Activities &a) const { return evaluate(a); }

	const Weights &weights() const { return m_weights; }

	Weights &weights() { return m_weights; }
};

/**
 * Structure defining the ReLU nonlinearity.
 */
struct ReLU {
	static double inverse(double a) { return a; }

	static double activation(double x) { return std::max(x, 0.0); }

	static double derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }
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

	static double activation(double x)
	{
		if (x >= 1.0 + 1e-6) {
			return 1.0 / (slope - std::log1p(-1.0 / x));
		}
		return 0.0;
	}

	static double derivative(double x)
	{
		if (x >= 1.0 + 1e-6) {
			const double a = activation(x);
			return (a * a) / (x * x * (1.0 - 1.0 / x));
		}
		return 0.0;
	}
};

/**
 * Class implementing a single neural ensemble.
 */
template <typename NEURON_TYPE, size_t N_NEURONS, size_t N_DIMS>
class Ensemble {
public:
	static constexpr size_t n_neurons = N_NEURONS;
	static constexpr size_t n_dims = N_DIMS;

	using NeuronVec = Eigen::Array<double, N_NEURONS, 1>;
	using Input = Eigen::Matrix<double, N_DIMS, 1>;
	using Activities = Eigen::Matrix<double, N_NEURONS, 1>;
	using Gain = NeuronVec;
	using Bias = NeuronVec;
	using Encoder =
	    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

private:
	Gain m_gain;
	Bias m_bias;
	Encoder m_encoder;

	void compute_gain_bias(std::mt19937 &rng)
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
	}

public:
	Ensemble(std::mt19937 &rng)
	{
		compute_gain_bias(rng);
		m_encoder = Dists::hypersphere(n_neurons, n_dims, rng);
	}

	Activities activities(const Input &input) const
	{
		Activities res;
		for (size_t i = 0; i < n_neurons; i++) {
			res[i] = NEURON_TYPE::activation(
			    m_gain[i] * m_encoder.row(i) * input + m_bias[i]);
		}
		return res;
	}

	Activities derivative(const Input &input) const
	{
		Activities res;
		for (size_t i = 0; i < n_neurons; i++) {
			res[i] = m_gain[i] *
			         NEURON_TYPE::derivative(
			             m_gain[i] * m_encoder.row(i) * input + m_bias[i]);
		}
		return res;
	}

	Activities operator()(const Input &input) const
	{
		return activities(input);
	}

	const Encoder &encoder() const { return m_encoder; }
	Encoder &encoder() { return m_encoder; }

	const Gain &gain() const { return m_gain; }
	Gain &gain() { return m_gain; }

	const Bias &bias() const { return m_bias; }
	Bias &bias() { return m_bias; }
};
