use na::{DMat, Transpose, ColSlice, Shape};
use std::ops::{Div, Index, IndexMut};
use std;
use nalgebra;

const MAX_ITER: usize = 1000;

pub struct LinearRegression {
	/// Scaled data.
	prices: DMat<f64>,
	/// Scaled data.
	mileages: DMat<f64>,
    nb_iter: usize,
    nrows: f64,
    learning_rate: f64,

    /// The value which will decide when the gradient descent will stop.
    /// If between two iteration the cost will decrease to less than threshold,
    /// the gradient descent will stop
    threshold: f64,
    thetas: (f64, f64),
    end_cost: f64
}

impl LinearRegression {
	fn one_loop(&self, ab: &DMat<f64>) -> (f64, f64, f64) {
		let cost = (self.mileages.clone() * &ab.transpose()) - self.prices.clone();
		let cost_mul_mileage = cost.transpose() * self.mileages.col_slice(1, 0, self.mileages.nrows());
		let ttl_cost = nalgebra::sum_mat_cells(&cost).abs();
		let ttl_cost_mul_mileage = nalgebra::sum_vec_cells(&cost_mul_mileage).abs();
		let avg_cost = ttl_cost / self.nrows;
		println!("###iter {} cost {} i0 {} i1 {}", self.nb_iter, avg_cost, ab.index((0, 0)), ab.index((0, 1)));
		let tmp0 = self.learning_rate * ttl_cost / self.nrows;
		let tmp1 = self.learning_rate * ttl_cost_mul_mileage / self.nrows;
		(tmp0, tmp1, avg_cost)
	}

	/// return (theta0, theta1)
	fn gradient_descent(&mut self) {
		//first is alpha, second is beta
		let mut ab = DMat::new_zeros(1, 2);

		// iterate to update parameters.
		let mut previous_cost = std::f64::MAX;
		println!("self.nrows {:?}", self.nrows);
		loop {
			let (theta0, theta1, avg_cost) = self.one_loop(&ab);
			// if avg_cost > previous_cost {
			// 	panic!("Error: the cost is raising after one iteration. Change the learning rate to fix it.");
			// }
			// if previous_cost - avg_cost < self.threshold {
			// 	println!("The decrease of cost is below threshold {}. Stop gradient descent after {} iteration.", self.threshold, self.nb_iter);
			//     break ;
			// }
			if self.nb_iter > MAX_ITER {
				break ;
			}
			*ab.index_mut((0, 0)) = theta0;
			*ab.index_mut((0, 1)) = theta1;
			// println!("mileages {:?} ab {:?} prices {:?}", self.mileages.shape(), ab.transpose().shape(), self.prices.shape());
			self.nb_iter += 1;
			previous_cost = avg_cost;
		}
		self.thetas = (*ab.index((0, 0)), *ab.index((0, 1)));
	}

	fn scale_fn(mat: &DMat<f64>, nrows: f64) -> DMat<f64> {
		if nalgebra::avg_mat(mat) == 0. {
			return mat.clone();
		}
		mat.clone().div(nalgebra::avg_mat(mat))
	}

	fn scale(mileages: &DMat<f64>, prices:&DMat<f64>) -> (DMat<f64>, DMat<f64>) {
		let nrows = mileages.nrows() as f64;
		(LinearRegression::scale_fn(mileages, nrows),
				LinearRegression::scale_fn(prices, nrows))
	}

	pub fn new(
		mileages: &DMat<f64>,
		prices: &DMat<f64>,
		learning_rate: f64,
		threshold: f64
	) -> LinearRegression {
		let nrows = mileages.nrows() as f64;
	    let (mileages_scaled, prices_scaled) = LinearRegression::scale(mileages, prices);
	    // adding column of 1. to the mileage mat
	    let mileages_expanded = nalgebra::add_col_one(&mileages_scaled);
		println!("{:?}{:?}", mileages_expanded, prices_scaled);
		let mut ln = LinearRegression {
			prices: prices_scaled,
			mileages: mileages_expanded,
		    nb_iter: 0,
		    nrows: nrows,
		    learning_rate: learning_rate,
		    threshold: threshold,
		    thetas: (0., 0.),
		    end_cost: 0.
		};
		ln.gradient_descent();
		ln
	}

	pub fn estimate_price(&self, mileage: f64) -> f64 {
		let (theta0, theta1) = self.thetas;
		theta0 + theta1 * mileage
	}

	pub fn get_thetas(&self) -> (f64, f64) {
	    self.thetas
	}
}

#[cfg(test)]
mod test
{
	use super::*;
	use na::{DMat};

	fn one_test(
		mileages: Vec<f64>,
		prices: Vec<f64>,
		learning_rate: f64,
		threshold: f64,
		expected: (f64, f64)
	) {
		let mat_mileages = DMat::from_col_vec(mileages.len(), 1, &*mileages.into_boxed_slice());
		let mat_prices = DMat::from_col_vec(prices.len(), 1, &*prices.into_boxed_slice());
		let ln = LinearRegression::new(&mat_mileages, &mat_prices, learning_rate, threshold);
		assert!(expected == ln.get_thetas());
	}

	#[test]
	fn test_one_loop() {
		one_test(vec!(0.), vec!(1.), 1., 0.00000001, (1., 0.));
		one_test(vec!(3., 2., 1.), vec!(1., 2., 3.), 1.5, 0.00000001, (1., 0.));
	}
}
