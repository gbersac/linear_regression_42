use na::{DMat, Transpose, ColSlice, Shape};
use std::ops::{Div, Index, IndexMut};
use std;
use nalgebra;

const LEARNING_RATE: f64 = 0.5;
const THRESHOLD: f64 = 0.01;

pub struct LinearRegression {
	/// Scaled data.
	prices: DMat<f64>,
	/// Scaled data.
	mileages: DMat<f64>,
    nb_iter: usize,
    nrows: f64,
    learning_rate: f64,
    thetas: (f64, f64)
}

impl LinearRegression {
	/// return (theta0, theta1)
	fn gradient_descent(&mut self) {
		//first is alpha, second is beta
		let mut ab = DMat::new_zeros(1, 2);

		// iterate to update parameters.
		let mut previous_cost = std::f64::MAX;
		println!("self.nrows {:?}", self.nrows);
		loop {
			// println!("mileages {:?} ab {:?} prices {:?}", self.mileages.shape(), ab.transpose().shape(), self.prices.shape());
			let cost = (self.mileages.clone() * &ab.transpose()) - self.prices.clone();
			let cost_mul_mileage = cost.transpose() * self.mileages.col_slice(1, 0, self.mileages.nrows());
			let ttl_cost = nalgebra::sum_mat_cells(&cost).abs();
			let ttl_cost_mul_mileage = nalgebra::sum_vec_cells(&cost_mul_mileage).abs();
			let avg_cost = ttl_cost / self.nrows;
			println!("###iter {} cost {} i0 {} i1 {}", self.nb_iter, avg_cost, ab.index((0, 0)), ab.index((0, 1)));
			if avg_cost > previous_cost {
				panic!("Error: the cost is raising after one iteration. Change the learning rate to fix it.");
			}
			if previous_cost - avg_cost < THRESHOLD {
				println!("The decrease of cost is beyond threshold {}. Stop gradient descent after {} iteration.", THRESHOLD, self.nb_iter);
			    break ;
			}
			let tmp0 = self.learning_rate * ttl_cost / self.nrows;
			let tmp1 = self.learning_rate * ttl_cost_mul_mileage / self.nrows;
			*ab.index_mut((0, 0)) = tmp0;
			*ab.index_mut((0, 1)) = tmp1;
			self.nb_iter += 1;
			previous_cost = avg_cost;
		}
		self.thetas = (*ab.index((0, 0)), *ab.index((0, 1)));
	}

	fn scale_fn(mat: &DMat<f64>, nrows: f64) -> DMat<f64> {
		let avg = nalgebra::sum_mat_cells(&mat) / nrows;
		mat.clone().div(avg)
	}

	fn scale(mileages: &DMat<f64>, prices:&DMat<f64>) -> (DMat<f64>, DMat<f64>) {
		let nrows = mileages.nrows() as f64;
		(LinearRegression::scale_fn(mileages, nrows),
				LinearRegression::scale_fn(prices, nrows))
	}

	pub fn new(mileages: &DMat<f64>, prices:&DMat<f64>) -> LinearRegression {
		let nrows = mileages.nrows() as f64;
	    let (mileages_scaled, prices_scaled) = LinearRegression::scale(mileages, prices);
	    // adding column of 1. to the mileage mat
	    let mileages_expanded = nalgebra::add_col_one(&mileages_scaled);
		// println!("{:?}{:?}", mileages_expanded, prices_scaled);
		let mut ln = LinearRegression {
			prices: prices_scaled,
			mileages: mileages_expanded,
		    nb_iter: 0,
		    nrows: nrows,
		    learning_rate: LEARNING_RATE,
		    thetas: (0., 0.)
		};
		ln.gradient_descent();
		ln
	}

	pub fn estimate_price(&self, mileage: f64) -> f64 {
		let (theta0, theta1) = self.thetas;
		theta0 + theta1 * mileage
	}
}
