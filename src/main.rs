extern crate csv;

use std::io::prelude::*;
use std::fs::File;

type Row = (i32, i32);

fn decode_one_file(file_name: &String) -> Result<Vec<Row>, std::io::Error> {
	let mut f = try!(File::open(file_name));
	let mut s = String::new();
	try!(f.read_to_string(&mut s));
	let mut rdr = csv::Reader::from_string(s);
	let to_return = rdr.decode().collect::<csv::Result<Vec<Row>>>().unwrap();
	Ok(to_return)
}

fn main() {
	let mut is_first = true;
	for arg in std::env::args() {
		if is_first {
			is_first = false;
			continue ;
		}
		let s = decode_one_file(&arg);
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
