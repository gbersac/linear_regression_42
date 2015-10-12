extern crate nalgebra as na;
extern crate csv;

mod car;

use std::io::prelude::*;
use std::fs::File;
use car::{Car};
use std::ops::IndexMut;
use na::{DMat};

type Row = (i32, i32);

fn decode_one_file(file_name: &String) -> Result<DMat<i32>, std::io::Error> {
	let mut f = try!(File::open(file_name));
	let mut s = String::new();
	try!(f.read_to_string(&mut s));
	let mut rdr = csv::Reader::from_string(s);
	let rows = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
	let mut to_return = DMat::new_zeros(rows.len(), 2);
	for (i, row) in rows.iter().enumerate() {
		*to_return.index_mut((i, 0)) = row.0;
		*to_return.index_mut((i, 1)) = row.1;
	}
	Ok(to_return)
}

fn main() {
	let mut is_first = true;
	for arg in std::env::args() {
		if is_first {
			is_first = false;
			continue ;
		}
		let mut s = decode_one_file(&arg);
		match s {
			Ok(s) => {
				println!("{:?}", s);
			},
			Err(e) => {
				println!("Cannot open file {}. Error {}", arg, e);
			}
		}
	}
}
