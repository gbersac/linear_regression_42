use na::{DMat, DVec};
use std::ops::{Index, IndexMut};

/// Return the sums of all the cells of a matrix
pub fn sum_mat_cells(mat: &DMat<f64>) -> f64
{
	let mut ttl = 0.;
	for x in 0..mat.ncols() {
		for y in 0..mat.nrows() {
			ttl = ttl + *mat.index((y, x));
		}
	}
	ttl
}

/// Return the sums of all the cells of a vector
pub fn sum_vec_cells(vec: &DVec<f64>) -> f64
{
	let mut ttl = 0.;
	for x in 0..vec.len() {
		ttl += *vec.index(x)
	}
	ttl
}

/// Add a column of 1. on the left col of the matrix
pub fn add_col_one(mat: &DMat<f64>) -> DMat<f64> {
	let mut to_return = DMat::new_ones(mat.nrows(), mat.ncols() + 1);
	for y in 0..mat.nrows() {
		for x in 0..mat.ncols() {
			*to_return.index_mut((y, x + 1)) = *mat.index((y, x));
		}
	}
	to_return
}

pub fn avg_mat(mat: &DMat<f64>) -> f64 {
	sum_mat_cells(mat) / (mat.nrows() as f64)
}
