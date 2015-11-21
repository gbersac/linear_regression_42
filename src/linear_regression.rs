use na::{DMat, Transpose, ColSlice, Shape};
use std::ops::{Div, Index, IndexMut};
use std;
use nalgebra;

const MAX_ITER: usize = 10000;

pub struct LinearRegression {
	/// Scaled data.
	prices: DMat<f64>,
	/// Scaled data.
	mileages: DMat<f64>,
    nb_iter: usize,
    nrows: f64,
    learning_rate: f64,

    /// This is the average used to scale datas.
    /// First is the avg used to scale mileage.
    /// Second is the avg used to scale prices.
    avgs: (f64, f64),

    /// The value which will decide when the gradient descent will stop.
    /// If between two iteration the cost will decrease to less than threshold,
    /// the gradient descent will stop
    threshold: f64,
    thetas: (f64, f64),
    end_cost: f64
}

impl LinearRegression {
	fn one_loop(&self, ab: &DMat<f64>) -> (f64, f64, f64) {
		let mut ttl0 = 0.;
		let mut ttl1 = 0.;

		let theta0 = ab.index((0, 0));
		let theta1 = ab.index((0, 1));
		let mut ttl_cost = 0.;
		for i in 0..self.mileages.nrows() {
			let mileage = self.mileages.index((i, 1));
			// println!("mileage {:?} price {:?}", mileage, self.prices.index((i, 0)));
			let cost = (theta0 + theta1 * mileage) - self.prices.index((i, 0));
			ttl0 += cost;
			ttl1 += cost * mileage;
			ttl_cost += cost;
		}
		// println!("###iter {} cost {} i0 {} i1 {}", self.nb_iter, ttl_cost/ self.nrows, ab.index((0, 0)), ab.index((0, 1)));
		let tmp0 = theta0 - self.learning_rate * (ttl0 / self.nrows);
		let tmp1 = theta1 - self.learning_rate * (ttl1 / self.nrows);
		(tmp0, tmp1, ttl_cost / self.nrows)
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
			self.end_cost = avg_cost;
			// if avg_cost.abs() > previous_cost.abs() {
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

	fn scale_fn(mat: &DMat<f64>, nrows: f64) -> (DMat<f64>, f64) {
		let avg = nalgebra::avg_mat(mat);
		if nalgebra::avg_mat(mat) == 0. {
			return (mat.clone(), 1.);
		}
		(mat.clone().div(avg), avg)
	}

	fn scale(
		mileages: &DMat<f64>,
		prices: &DMat<f64>
	) -> ((DMat<f64>, DMat<f64>), (f64, f64)) {
		let nrows = mileages.nrows() as f64;
		let (mileages_scaled, mileages_avg) = LinearRegression::scale_fn(mileages, nrows);
		let (prices_scaled, prices_avg) = LinearRegression::scale_fn(prices, nrows);
		((mileages_scaled, prices_scaled), (mileages_avg, prices_avg))
	}

	pub fn new(
		mileages: &DMat<f64>,
		prices: &DMat<f64>,
		learning_rate: f64,
		threshold: f64
	) -> LinearRegression {
		let nrows = mileages.nrows() as f64;
	    let ((mileages_scaled, prices_scaled), avgs) = LinearRegression::scale(mileages, prices);

	    // let (mileages_scaled, prices_scaled) = (mileages.clone(), prices.clone());
	    // let avgs = (1., 1.);

	    // adding column of 1. to the mileage mat
	    let mileages_expanded = nalgebra::add_col_one(&mileages_scaled);
		// println!("{:?}{:?}", mileages_expanded, prices_scaled);
		let mut ln = LinearRegression {
			prices: prices_scaled,
			mileages: mileages_expanded,
		    nb_iter: 0,
		    nrows: nrows,
		    learning_rate: learning_rate,
		    threshold: threshold,
		    thetas: (0., 0.),
		    avgs: avgs,
		    end_cost: 0.
		};
		ln.gradient_descent();
		ln
	}

	pub fn estimate_price(&self, mileage: f64) -> f64 {
		let (theta0, theta1) = self.thetas;
		theta0 + theta1 * mileage
	}

	pub fn get_end_cost(&self) -> f64 {
		let (_, prices_avg) = self.avgs;
	    (self.end_cost * prices_avg).abs()
	}

	pub fn get_thetas(&self) -> (f64, f64) {
		let (theta0, theta1) = self.thetas;
		let (_, prices_avg) = self.avgs;
		(theta0 * prices_avg, theta1)
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
		let (theta0, theta1) = ln.get_thetas();
		let (exp0, exp1) = expected;
		println!("theta0 {:?} theta1 {}", theta0, theta1);
		assert!(exp0 == theta0.round() && (theta1 - exp1).abs() < 0.001);
	}

	#[test]
	fn test_one_loop() {
		one_test(vec!(0.), vec!(1.), 1., 0.00000001, (1., 0.));
		one_test(vec!(3., 2., 1.), vec!(1., 2., 3.), 0.1, 0.00000001, (4., -1.));
		one_test(
			vec!(240000.,139800.,150500.,185530.,176000.,114800.,166800.,89000.,144500.,84000.,82029.,63060.,74000.,97500.,67000.,76025.,48235.,93000.,60949.,65674.,54000.,68500.,22899.,61789.),
			vec!(3650.,3800.,4400.,4450.,5250.,5350.,5800.,5990.,5999.,6200.,6390.,6390.,6600.,6800.,6800.,6900.,6900.,6990.,7490.,7555.,7990.,7990.,7990.,8290.),
			0.1,
			0.001,
			(8500., -0.3423)
		)
	}
}
