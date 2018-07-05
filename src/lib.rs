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
//!    assert_eq!(Ok((0.6, 2.2)), linear_regression(&xs, &ys));
//!
//!
//!    // Example 2: x and y values stored as tuples
//!    let tuples: Vec<(f32, f32)> = vec![(1.0, 2.0),
//!                                       (2.0, 4.0),
//!                                       (3.0, 5.0),
//!                                       (4.0, 4.0),
//!                                       (5.0, 5.0)];
//!
//!    assert_eq!(Ok((0.6, 2.2)), linear_regression_of(&tuples));
//!
//!
//!    // Example 3: directly operating on integer (converted to float as required)
//!    let xs: Vec<u8> = vec![1, 2, 3, 4, 5];
//!    let ys: Vec<u8> = vec![2, 4, 5, 4, 5];
//!
//!    assert_eq!(Ok((0.6, 2.2)), linear_regression(&xs, &ys));
//! ```
#![no_std]

extern crate num_traits;

use num_traits::float::FloatCore;

#[cfg(test)]
#[macro_use]
extern crate std;

use core::fmt;
use core::iter::Iterator;
use core::iter::Sum;

/// The kinds of errors that can occur when calculating a linear regression.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Error {
    /// Tried to divide by Zero.
    DivByZero,
    /// Lengths of the inputs are different.
    InputLenDif(usize, usize),
    /// Converting to a [Float](../num_traits/float/trait.FloatCore.html) failed.
    FloatConvError(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let description = match self {
            Error::DivByZero => "Tried to divide by zero",
            Error::InputLenDif(_, _) => "Lengths of inputs are different",
            Error::FloatConvError(_) => "Failed to convert into a Float",
        };

        description.fmt(f)
    }
}

/// Calculates a linear regression.
///
/// Lower-level linear regression function. Assumes that `x_mean` and `y_mean`
/// have already been calculated. Returns `Error::DivByZero` if
///
/// * the slope is too steep to represent, approaching infinity.
///
/// Since there is a mean, this function assumes that `xs` and `ys` are both non-empty.
///
/// Returns `Some(slope, intercept)` of the regression line.
pub fn lin_reg<'a, I, F>(xys: I, x_mean: F, y_mean: F) -> Result<(F, F), Error>
where
    I: Iterator<Item = (F, F)>,
    F: FloatCore,
{
    // SUM (x-mean(x))^2
    let mut xxm2 = F::zero();

    // SUM (x-mean(x)) (y-mean(y))
    let mut xmym2 = F::zero();

    for (x, y) in xys {
        xxm2 = xxm2 + (x - x_mean) * (x - x_mean);
        xmym2 = xmym2 + (x - x_mean) * (y - y_mean);
    }

    let slope = xmym2 / xxm2;

    // we check for divide-by-zero after the fact
    if slope.is_nan() {
        return Err(Error::DivByZero);
    }

    let intercept = y_mean - slope * x_mean;

    Ok((slope, intercept))
}

/// Linear regression from two slices.
///
/// Calculates the linear regression from two slices, one for x- and one for y-values.
///
/// Returns an error if
///
/// * `xs` and `ys` differ in length
/// * `xs` or `ys` are empty
/// * the slope is too steep to represent, approaching infinity
/// * the number of elements cannot be represented as an `F`
///
/// Returns `Ok(slope, intercept)` of the regression line.
pub fn linear_regression<X, Y, F>(xs: &[X], ys: &[Y]) -> Result<(F, F), Error>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: FloatCore + Sum,
{
    if xs.len() != ys.len() {
        return Err(Error::InputLenDif(xs.len(), ys.len()));
    }

    if xs.is_empty() {
        return Err(Error::DivByZero.into());
    }
    let x_sum: F = xs.iter().cloned().map(|i| i.into()).sum();
    let n = F::from(xs.len()).ok_or(Error::FloatConvError(xs.len()))?;
    let x_mean = x_sum / n;
    let y_sum: F = ys.iter().cloned().map(|i| i.into()).sum();
    let y_mean = y_sum / n;

    lin_reg(
        xs.iter()
            .map(|i| i.clone().into())
            .zip(ys.iter().map(|i| i.clone().into())),
        x_mean,
        y_mean,
    )
}

/// Linear regression from tuples.
///
/// Calculates the linear regression from a slice of tuple values.
///
/// Returns an error if
///
/// * `xys` is empty
/// * the slope is too steep to represent, approaching infinity
/// * the number of elements cannot be represented as an `F`
///
/// Returns `Ok(slope, intercept)` of the regression line.
pub fn linear_regression_of<X, Y, F>(xys: &[(X, Y)]) -> Result<(F, F), Error>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: FloatCore,
{
    if xys.is_empty() {
        return Err(Error::DivByZero.into());
    }
    // We're handrolling the mean computation here, because our generic implementation can't handle tuples.
    // If we ran the generic impl on each tuple field, that would be very cache inefficient
    let n = F::from(xys.len()).ok_or(Error::FloatConvError(xys.len()))?;
    let (x_sum, y_sum) = xys
        .iter()
        .cloned()
        .fold((F::zero(), F::zero()), |(sx, sy), (x, y)| {
            (sx + x.into(), sy + y.into())
        });
    let x_mean = x_sum / n;
    let y_mean = y_sum / n;

    lin_reg(
        xys.iter()
            .map(|(x, y)| (x.clone().into(), y.clone().into())),
        x_mean,
        y_mean,
    )
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;

    #[test]
    fn float_slices_regression() {
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];

        assert_eq!(Ok((0.6, 2.2)), linear_regression(&xs, &ys));
    }

    #[test]
    fn int_slices_regression() {
        let xs: Vec<u8> = vec![1, 2, 3, 4, 5];
        let ys: Vec<u8> = vec![2, 4, 5, 4, 5];

        assert_eq!(Ok((0.6, 2.2)), linear_regression(&xs, &ys));
    }

    #[test]
    fn float_tuples_regression() {
        let tuples: Vec<(f32, f32)> =
            vec![(1.0, 2.0), (2.0, 4.0), (3.0, 5.0), (4.0, 4.0), (5.0, 5.0)];

        assert_eq!(Ok((0.6, 2.2)), linear_regression_of(&tuples));
    }

    #[test]
    fn int_tuples_regression() {
        let tuples: Vec<(u32, u32)> = vec![(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)];

        assert_eq!(Ok((0.6, 2.2)), linear_regression_of(&tuples));
    }
}
