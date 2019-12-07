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
#![warn(clippy::pedantic)]

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
    /// The slope is too steep to represent, approaching infinity.
    TooSteep,
    /// Failed to calculate mean.
    /// This means the input was empty or had too many elements.
    Mean,
    /// Lengths of the inputs are different.
    InputLenDif,
    /// Can't compute linear regression of zero elements
    NoElements,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::TooSteep => write!(
                f,
                "The slope is too steep to represent, approaching infinity."
            ),
            Self::Mean => write!(
                f,
                "Failed to calculate mean. Input was empty or had too many elements"
            ),
            Self::InputLenDif => write!(f, "Lengths of the inputs are different"),
            Self::NoElements => write!(f, "Can't compute linear regression of zero elements"),
        }
    }
}

/// Calculates a linear regression without requiring pre-computation of the mean.
///
/// Lower-level linear regression function.
/// A bit less precise than `linreg`, but mostly irrelevant in practice.
///
/// Errors if the number of elements is too large to be represented as `F` or
/// the slope is too steep to represent, approaching infinity.
///
/// Returns `Ok((slope, intercept))` of the regression line.
pub fn lin_reg_imprecise<I, F>(xys: I) -> Result<(F, F), Error>
where
    F: FloatCore,
    I: Iterator<Item = (F, F)>,
{
    details::lin_reg_imprecise_components(xys)?.finish()
}

/// A module containing the building parts of the main API.
/// You can use these if you want to have more control over the linear regression
pub mod details {
    use super::Error;
    use num_traits::float::FloatCore;

    /// Low level linear regression primitive for pushing values instead of fetching them
    /// from an iterator
    pub struct Accumulator<F: FloatCore> {
        x_mean: F,
        y_mean: F,
        x_mul_y_mean: F,
        x_squared_mean: F,
        n: usize,
    }

    impl<F: FloatCore> Default for Accumulator<F> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<F: FloatCore> Accumulator<F> {
        pub fn new() -> Self {
            Self {
                x_mean: F::zero(),
                y_mean: F::zero(),
                x_mul_y_mean: F::zero(),
                x_squared_mean: F::zero(),
                n: 0,
            }
        }

        pub fn push(&mut self, x: F, y: F) {
            self.x_mean = self.x_mean + x;
            self.y_mean = self.y_mean + y;
            self.x_mul_y_mean = self.x_mul_y_mean + x * y;
            self.x_squared_mean = self.x_squared_mean + x * x;
            self.n += 1;
        }

        pub fn normalize(&mut self) -> Result<(), Error> {
            match self.n {
                1 => return Ok(()),
                0 => return Err(Error::NoElements),
                _ => {}
            }
            let n = F::from(self.n).ok_or(Error::Mean)?;
            self.n = 1;
            self.x_mean = self.x_mean / n;
            self.y_mean = self.y_mean / n;
            self.x_mul_y_mean = self.x_mul_y_mean / n;
            self.x_squared_mean = self.x_squared_mean / n;
            Ok(())
        }

        pub fn parts(mut self) -> Result<(F, F, F, F), Error> {
            self.normalize()?;
            let Self {
                x_mean,
                y_mean,
                x_mul_y_mean,
                x_squared_mean,
                ..
            } = self;
            Ok((x_mean, y_mean, x_mul_y_mean, x_squared_mean))
        }

        pub fn finish(self) -> Result<(F, F), Error> {
            let (x_mean, y_mean, x_mul_y_mean, x_squared_mean) = self.parts()?;
            let slope = (x_mul_y_mean - x_mean * y_mean) / (x_squared_mean - x_mean * x_mean);
            let intercept = y_mean - slope * x_mean;

            if slope.is_nan() {
                return Err(Error::TooSteep);
            }

            Ok((slope, intercept))
        }
    }

    pub fn lin_reg_imprecise_components<I, F>(xys: I) -> Result<Accumulator<F>, Error>
    where
        F: FloatCore,
        I: Iterator<Item = (F, F)>,
    {
        let mut acc = Accumulator::new();

        for (x, y) in xys {
            acc.push(x, y);
        }

        acc.normalize()?;
        Ok(acc)
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
/// Returns `Ok((slope, intercept))` of the regression line.
pub fn lin_reg<I, F>(xys: I, x_mean: F, y_mean: F) -> Result<(F, F), Error>
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
        return Err(Error::TooSteep);
    }

    let intercept = y_mean - slope * x_mean;

    Ok((slope, intercept))
}

/// Linear regression from two slices.
///
/// Calculates the linear regression from two slices, one for x- and one for y-values.
/// This requires two iterations over the slices in order to precompute the mean. For
/// large slices it may be faster to use `lin_reg_imprecise` instead.
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
        return Err(Error::InputLenDif);
    }

    if xs.is_empty() {
        return Err(Error::Mean);
    }
    let x_sum: F = xs.iter().cloned().map(Into::into).sum();
    let n = F::from(xs.len()).ok_or(Error::Mean)?;
    let x_mean = x_sum / n;
    let y_sum: F = ys.iter().cloned().map(Into::into).sum();
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
/// This requires two iterations over the slice in order to precompute the mean. For
/// large slices it may be faster to use `lin_reg_imprecise` instead.
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
        return Err(Error::Mean);
    }
    // We're handrolling the mean computation here, because our generic implementation can't handle tuples.
    // If we ran the generic impl on each tuple field, that would be very cache inefficient
    let n = F::from(xys.len()).ok_or(Error::Mean)?;
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
    fn lin_reg_imprecises_vs_linreg() {
        let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];

        let (x1, y1) = lin_reg_imprecise(xs.iter().cloned().zip(ys.iter().cloned())).unwrap();
        let (x2, y2): (f64, f64) = linear_regression(&xs, &ys).unwrap();

        assert!(f64::abs(x1 - x2) < 0.00001);
        assert!(f64::abs(y1 - y2) < 0.00001);
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
