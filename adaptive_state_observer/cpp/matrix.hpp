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

static constexpr int layout(int M, int N)
{
	return Eigen::AutoAlign | (((M != 1) && (N == 1)) ? 0 : Eigen::RowMajor);
}

template <int M, int N>
using Mat = Eigen::Matrix<double, M, N, layout(M, N)>;

template <int M>
using Vec = Eigen::Matrix<double, M, 1>;

template <int M, int N = 1>
using Arr = Eigen::Array<double, M, N, layout(M, N)>;

template <typename T, int N>
using Stride = Eigen::Stride<(T::Options & Eigen::RowMajor) ? N : 1,
                             (T::Options & Eigen::RowMajor) ? 1 : N>;

