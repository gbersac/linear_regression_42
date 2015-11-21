extern crate nalgebra as na;
extern crate csv;

const LEARNING_RATE: f64 = 0.5;
const THRESHOLD: f64 = 0.001;

mod linear_regression;
mod nalgebra;

use std::io::prelude::*;
use std::fs::File;
use na::{DMat};
use std::ops::{IndexMut};
use linear_regression::{LinearRegression};

type Row = (f64, f64);

fn decode_one_file(file_name: &String)
		-> Result<(DMat<f64>, DMat<f64>), std::io::Error> {
	let mut f = try!(File::open(file_name));
	let mut s = String::new();
	try!(f.read_to_string(&mut s));
	let mut rdr = csv::Reader::from_string(s);
	let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
	let mut mileage = DMat::new_ones(rows.len(), 1);
	let mut prices = DMat::new_zeros(rows.len(), 1);
	for (i, row) in rows.iter().enumerate() {
		*mileage.index_mut((i, 0)) = row.0;
		*prices.index_mut((i, 0)) = row.1;
	}
	Ok((mileage, prices))
}

fn usage() {
    println!("Usage: linear_regression [file]+");
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
		let res = decode_one_file(&arg);
		match res {
			Ok(s) => {
				let (mileage, prices) = s;
				let ln = LinearRegression::new(&mileage, &prices, LEARNING_RATE, THRESHOLD);
				println!("result {:?}, average cost {}", ln.get_thetas(), ln.get_end_cost());
			},
			Err(e) => {
				println!("Cannot open file {}. Error {}", arg, e);
			}
		}
	}
}
