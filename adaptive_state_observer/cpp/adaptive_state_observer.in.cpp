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

#include <cstdint>
#include <cstddef>
#include <random>
#include <tuple>

#include <Eigen/Dense>

#include "observer.hpp"

using Observer =
    AdaptiveStateObserver<$N_STATE_DIM,       /* N_STATE_DIM */
                          $N_AUX_STATE_DIM,   /* N_AUX_STATE_DIM */
                          $N_OBSERVATION_DIM, /* N_OBSERVATION_DIM */
                          $N_CONTROL_DIM,     /* N_CONTROL_DIM */
                          $N_NEURONS_G,       /* N_NEURONS_G */
                          $N_NEURONS_H,       /* N_NEURONS_H */
                          $NEURON_TYPE        /* NEURON_TYPE */
                          >;

using Params = typename Observer::Params;

extern "C" {
Observer *aso_create(uint32_t seed)
{
	std::mt19937 rng(seed);
	return new Observer(rng);
}

void aso_free(Observer *observer) { delete observer; }

void aso_step_single(Observer *observer, const double *x, const double *u,
                     const double *z, const Params *p, double *x_out, double *z_out)
{
	Eigen::Map<const Observer::State> map_x(x);
	Eigen::Map<const Observer::Control> map_u(u);
	Eigen::Map<const Observer::Observation> map_z(z);
	Eigen::Map<Observer::State> map_x_out(x_out);
	Eigen::Map<Observer::Observation> map_z_out(z_out);

	std::tie(map_x_out, map_z_out) = observer->step(map_x, map_u, map_z, *p);
}

void aso_pred_dx_single(Observer *observer, const double *x, const double *u,
                        double *dx)
{
	Eigen::Map<const Observer::State> map_x(x);
	Eigen::Map<const Observer::Control> map_u(u);
	Eigen::Map<Observer::State> map_dx(dx);

	map_dx = observer->pred_dx(map_x, map_u);
}

void aso_pred_z_single(Observer *observer, const double *x, double *z)
{
	Eigen::Map<const Observer::State> map_x(x);
	Eigen::Map<Observer::Observation> map_z(z);

	map_z = observer->pred_z(map_x);
}

void aso_step(Observer *observer, const double *x0, const double *u,
              const double *z, const Params *p, double *x_out, double *z_out,
              uint32_t n)
{
	// Write to some temporary pointers on the stack in case no target pointer
	// was given
	Observer::State x_tmp;
	Observer::Observation z_tmp;
	const size_t x_out_stride = x_out ? $N_STATE_DIM : 0;
	const size_t z_out_stride = z_out ? $N_OBSERVATION_DIM : 0;
	x_out = x_out ? x_out : x_tmp.data();
	z_out = z_out ? z_out : z_tmp.data();

	for (uint32_t i = 0; i < n; i++) {
		// Use "x0" for the first iteration; the last state for later iterations
		const double *x_in = (i == 0) ? x0 : (x_out + (x_out_stride * (i - 1)));

		// Compute the other input pointers
		const double *u_in = u + $N_CONTROL_DIM * i;
		const double *z_in = z + $N_OBSERVATION_DIM * i;

		// Compute the target pointers
		double *x_tar = x_out + (x_out_stride * i);
		double *z_tar = z_out + (z_out_stride * i);

		// Perform a single step
		aso_step_single(observer, x_in, u_in, z_in, p, x_tar, z_tar);
	}
}

void aso_pred_dx(Observer *observer, const double *x, const double *u,
                 double *dx, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++) {
		aso_pred_dx_single(observer, x + $N_STATE_DIM * i,
		                   u + $N_CONTROL_DIM * i, dx + $N_STATE_DIM * i);
	}
}

void aso_pred_z(Observer *observer, const double *x, double *z, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++) {
		aso_pred_z_single(observer, x + $N_STATE_DIM * i, z + $N_OBSERVATION_DIM * i);
	}
}

double *aso_get_G0(Observer *o) { return o->G0(); }

double *aso_get_H0(Observer *o) { return o->H0(); }

double *aso_get_g_weights(Observer *o) { return o->ens_g().weights().data(); }

double *aso_get_h_weights(Observer *o) { return o->ens_h().weights().data(); }

double *aso_get_g_params(Observer *o) { return o->ens_g().params().data(); }

double *aso_get_h_params(Observer *o) { return o->ens_h().params().data(); }
}
