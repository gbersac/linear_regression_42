extern crate nalgebra as na;
extern crate csv;

use std::io::prelude::*;
use std::fs::File;
use std::ops::IndexMut;
use na::{DMat, DVec, Transpose, Shape, ColSlice};
use std::ops::Index;
use std::ops::{Add, Div};

type Row = (f64, f64);

fn decode_one_file(file_name: &String)
		-> Result<(DMat<f64>, DMat<f64>), std::io::Error> {
	let mut f = try!(File::open(file_name));
	let mut s = String::new();
	try!(f.read_to_string(&mut s));
	let mut rdr = csv::Reader::from_string(s);
	let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
	let mut mileage = DMat::new_ones(rows.len(), 2);
	let mut prices = DMat::new_zeros(rows.len(), 1);
	for (i, row) in rows.iter().enumerate() {
		*mileage.index_mut((i, 1)) = row.0;
		*prices.index_mut((i, 0)) = row.1;
	}
	Ok((mileage, prices))
}

fn usage() {
    println!("Usage: linear_regression [file]+");
}

fn estimate_price(mileage: f64, theta0: f64, theta1: f64) -> f64 {
	theta0 + theta1 * mileage
}

/// Return the sums of all the cells of a matrix
fn sum_mat_cells(mat: &DMat<f64>) -> f64
{
	let mut ttl = 0.;
	for x in 0..mat.ncols() {
		for y in 0..mat.nrows() {
			ttl = ttl + *mat.index((y, x));
		}
	}
	ttl
}

fn sum_vec_cells(vec: &DVec<f64>) -> f64
{
	let mut ttl = 0.;
	for x in 0..vec.len() {
		ttl += *vec.index(x)
	}
	ttl
}

/// return (theta0, theta1)
fn gradient_descent(
	mileage: &DMat<f64>,
	real_prices: &DMat<f64>,
	learning_rate: f64
) -> (f64, f64) {
	//first is alpha, second is beta
	let mut ab = DMat::new_zeros(1, 2);

	// iterate to update parameters.
	let nrows = mileage.nrows() as f64;
	println!("nrows {:?}", nrows);
	for i in 1..10 {
		let cost = (mileage * &ab.transpose()) - real_prices;
		let cost_mul_mileage = cost.transpose() * mileage.col_slice(1, 0, mileage.nrows());
		let ttl_cost = sum_mat_cells(&cost).abs();
		let ttl_cost_mul_mileage = sum_vec_cells(&cost_mul_mileage).abs();
		println!("###iter {} cost {} i0 {} i1 {}", i, ttl_cost, ab.index((0, 0)), ab.index((0, 1)));
		let tmp0 = learning_rate * ttl_cost / nrows;
		let tmp1 = learning_rate * ttl_cost_mul_mileage / nrows;
		*ab.index_mut((0, 0)) = tmp0;
		*ab.index_mut((0, 1)) = tmp1;
		// println!("mileage\n{:?}\nab\n{:?}\nestimated_prices\n{:?}",
		// 	mileage, ab, cost);
		// *ab.index_mut((0, 1)) = row.0;
	}
	(*ab.index((0, 0)), *ab.index((0, 1)))
}

fn scale(mat: &DMat<f64>, avg: f64) -> DMat<f64> {
	mat.clone().div(avg)
}

fn main() {
	let mut is_first = true;
	if std::env::args().len() == 1 {
		usage();
	}
	for arg in std::env::args() {
		if is_first {
			is_first = false;
			continue ;
		}
		let mut res = decode_one_file(&arg);
		match res {
			Ok(s) => {
				let (mut mileage, mut prices) = s;
				let nrows = mileage.nrows() as f64;
			    let avg_mileage = sum_mat_cells(&mileage) / nrows;
				let mileage = scale(&mileage, avg_mileage);
			    let avg_prices = sum_vec_cells(&mileage.col_slice(1, 0, nrows as usize)) / nrows;
				let prices = scale(&prices, avg_prices);
				gradient_descent(&mileage, &prices, 1.);
			},
			Err(e) => {
				println!("Cannot open file {}. Error {}", arg, e);
			}
		}
	}
}
