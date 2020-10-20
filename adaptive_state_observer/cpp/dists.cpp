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

#include "dists.hpp"

namespace Dists {
Eigen::MatrixXd uniform(double min, double max, size_t n_smpls, size_t n_dims,
                        std::mt19937 &rng)
{
	std::uniform_real_distribution<double> dist(min, max);
	Eigen::MatrixXd res(n_smpls, n_dims);
	for (size_t i = 0; i < n_smpls; i++) {
		for (size_t j = 0; j < n_dims; j++) {
			res(i, j) = dist(rng);
		}
	}
	return res;
}

Eigen::MatrixXd hypersphere(size_t n_smpls, size_t n_dims, std::mt19937 &rng)
{
	std::normal_distribution<double> dist(0.0, 1.0);
	Eigen::MatrixXd res(n_smpls, n_dims);
	for (size_t i = 0; i < n_smpls; i++) {
		for (size_t j = 0; j < n_dims; j++) {
			res(i, j) = dist(rng);
		}
		res.row(i) /= res.row(i).norm();
	}
	return res;
}

Eigen::MatrixXd halton(size_t n_smpls, size_t n_dims)
{
	// Chi, H., Mascagni, M., & Warnock, T. (2005).
	// On the optimal Halton sequence. Mathematics and Computers in Simulation,
	// 70(1), 9–21. https://doi.org/10.1016/j.matcom.2005.03.004

	static constexpr size_t N_DIMS_MAX = 50;

	static const int p[N_DIMS_MAX] = {
	    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,
	    43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101,
	    103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
	    173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229};

	static const int w[N_DIMS_MAX] = {
	    1,  2,  2,  5,   3,   7,  3,  10,  18,  11, 17, 5,   17,
	    26, 40, 14, 40,  44,  12, 31, 45,  70,  8,  38, 82,  8,
	    12, 38, 47, 70,  29,  57, 97, 110, 32,  48, 84, 124, 155,
	    26, 69, 83, 157, 171, 8,  32, 112, 205, 15, 31};

	// Fallback for n_dims larger than w
	// XXX: Compute p and w on-the-fly. Computing p is trivial, but I don't
	// understand the math behind w very well.
	if (n_dims > N_DIMS_MAX) {
		std::mt19937 rng(459819);
		return uniform(0.0, 1.0, n_smpls, n_dims, rng);
	}

	// Fill the result matrix
	Eigen::MatrixXd res(n_smpls, n_dims);
	for (int i = 0; i < int(n_dims); i++) {
		for (int n = 0; n < int(n_smpls); n++) {
			double x = 0.0, div = double(p[i]);
			// Represent the number j in the basis p[i] as j = b0 + b1 * p[i] +
			// b2 * p[i]^2 ...
			int j = n;
			while (j > 0) {
				// Current digit
				const int b = j % p[i];

				// Apply the linear congruent permutation proposed in the Chi et
				// al. paper
				const int bp = (b * w[i]) % p[i];

				// Sum the digit divided by the current divisor
				x += double(bp) / div;

				// Go to the next digit, increment the divisor
				j /= p[i];
				div *= double(p[i]);
			}
			res(n, i) = x;
		}
	}
	return res;
}

}  // namespace Dists

