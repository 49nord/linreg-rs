//! Linear regression
//!
//! `linreg` calculates linear regressions for two dimensional measurements.
//!
//! ```rust
//!    use linreg::{linear_regression, linear_regression_of};
//!
//!    // Example 1: x and y values stored in two different vectors
//!    let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!    let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];
//!
//!    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
//!
//!
//!    // Example 2: x and y values stored as tuples
//!    let tuples: Vec<(f32, f32)> = vec![(1.0, 2.0),
//!                                       (2.0, 4.0),
//!                                       (3.0, 5.0),
//!                                       (4.0, 4.0),
//!                                       (5.0, 5.0)];
//!
//!    assert_eq!(Some((0.6, 2.2)), linear_regression_of(&tuples));
//!
//!
//!    // Example 3: directly operating on integer (converted to float as required)
//!    let xs: Vec<u8> = vec![1, 2, 3, 4, 5];
//!    let ys: Vec<u8> = vec![2, 4, 5, 4, 5];
//!
//!    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
//! ```
#![no_std]

extern crate num_traits;

use num_traits::Float;

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
use std::vec::Vec;

use core::iter::Iterator;

/// Calculate a mean over an iterator
pub trait IteratorMean<F> {
    /// Calculates the mean value of all returned items of an iterator. Returns
    /// `None` if either no items are present or more items than can be counted
    /// by `F` (conversion from `usize` to  `F` is not possible).
    fn mean(&mut self) -> Option<F>;
}

impl<'a, T, I, F> IteratorMean<F> for I
where
    T: 'a + Into<F> + Clone,
    I: Iterator<Item = &'a T>,
    F: Float,
{
    fn mean(&mut self) -> Option<F> {
        let mut total = F::zero();
        let mut count: usize = 0;

        loop {
            if let Some(i) = self.next() {
                total = total + i.clone().into();
                count += 1;
            } else {
                break;
            }
        }

        if count > 0 {
            Some(total / F::from(count)?)
        } else {
            None
        }
    }
}

#[test]
fn simple_integer_mean() {
    let vals: Vec<u32> = vec![5, 8, 12, 17];
    assert_eq!(10.5, vals.iter().mean().unwrap());
}

#[test]
fn simple_float_mean() {
    let vals: Vec<f64> = vec![5.0, 8.0, 12.0, 17.0];
    assert_eq!(10.5, vals.iter().mean().unwrap());
}

#[test]
fn empty_set_has_no_mean() {
    let res: Option<f32> = Vec::<u16>::new().iter().mean();
    assert!(res.is_none());
}

/// Calculates a linear regression
///
/// Lower-level linear regression function. Assumes that `x_mean` and `y_mean`
/// have already been calculated. Returns `None` if
///
/// * the slope is too steep to represent, approaching infinity
///
/// Since there is a mean, this function assumes that `xs` and `ys` are both non-empty.
///
/// Returns `Some(slope, intercept)` of the regression line.
pub fn lin_reg<'a, X, Y, IX, IY, F>(xs: IX, ys: IY, x_mean: F, y_mean: F) -> Option<(F, F)>
where
    X: 'a + Into<F> + Clone,
    Y: 'a + Into<F> + Clone,
    IX: Iterator<Item = &'a X>,
    IY: Iterator<Item = &'a Y>,
    F: Float,
{
    // SUM (x-mean(x))^2
    let mut xxm2 = F::zero();

    // SUM (x-mean(x)) (y-mean(y))
    let mut xmym2 = F::zero();

    for (x, y) in xs.zip(ys) {
        let x: F = x.clone().into();
        let y: F = y.clone().into();

        xxm2 = xxm2 + (x - x_mean) * (x - x_mean);
        xmym2 = xmym2 + (x - x_mean) * (y - y_mean);
    }

    let slope = xmym2 / xxm2;

    // we check for divide-by-zero after the fact
    if slope.is_nan() {
        return None;
    }

    let intercept = y_mean - slope * x_mean;

    Some((slope, intercept))
}

/// Linear regression from two slices
///
/// Calculates the linear regression from two slices, one for x- and one for y-values.
///
/// Returns `None` if
///
/// * `xs` and `ys` differ in length
/// * `xs` or `ys` do not have a mean (e.g. if they are empty, see `IteratorMean` for details)
/// * the slope is too steep to represent, approaching infinity
///
/// Returns `Some(slope, intercept)` of the regression line.
pub fn linear_regression<X, Y, F>(xs: &[X], ys: &[Y]) -> Option<(F, F)>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: Float,
{
    if xs.len() != ys.len() {
        return None;
    }

    // if one of the axes is empty, we return `None`
    let x_mean = xs.iter().mean()?;
    let y_mean = ys.iter().mean()?;

    lin_reg(xs.iter(), ys.iter(), x_mean, y_mean)
}

/// Linear regression from tuples
///
/// Calculates the linear regression from a slice of tuple values.
///
/// Returns `None` if
///
/// * `x` or `y` tuple members do not have a mean (e.g. if they are empty, see `IteratorMean` for details)
/// * the slope is too steep to represent, approaching infinity
///
/// Returns `Some(slope, intercept)` of the regression line.
pub fn linear_regression_of<X, Y, F>(xys: &[(X, Y)]) -> Option<(F, F)>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: Float,
{
    // FIXME: cache penalty here, we should be calculating both means in a single step
    //        to avoid iterating twice
    let x_mean = xys.iter().map(|(x, _)| x).mean()?;
    let y_mean = xys.iter().map(|(_, y)| y).mean()?;

    lin_reg(
        xys.iter().map(|(x, _)| x),
        xys.iter().map(|(_, y)| y),
        x_mean,
        y_mean,
    )
}

#[test]
fn test_example_regression() {
    let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];

    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
}

#[test]
fn test_example_regression_of() {
    let tuples: Vec<(f32, f32)> = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 5.0), (4.0, 4.0), (5.0, 5.0)];

    assert_eq!(Some((0.6, 2.2)), linear_regression_of(&tuples));
}

#[test]
fn test_integer_regression() {
    let xs: Vec<u8> = vec![1, 2, 3, 4, 5];
    let ys: Vec<u8> = vec![2, 4, 5, 4, 5];

    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
}
