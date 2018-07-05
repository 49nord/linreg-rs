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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::TooSteep => write!(
                f,
                "The slope is too steep to represent, approaching infinity."
            ),
            Error::Mean => write!(
                f,
                "Failed to calculate mean. Input was empty or had too many elements"
            ),
            Error::InputLenDif => write!(f, "Lengths of the inputs are different"),
        }
    }
}

/// Calculate a mean over an iterator.
pub trait IteratorMean<F> {
    /// Calculates the mean value of all returned items of an iterator. Returns
    /// an error if either no items are present or more items than can be counted
    /// by `F` (conversion from `usize` to `F` is not possible).
    fn mean(&mut self) -> Option<F>;
}

impl<'a, T, I, F> IteratorMean<F> for I
where
    T: 'a + Into<F> + Clone,
    I: Iterator<Item = &'a T>,
    F: FloatCore,
{
    fn mean(&mut self) -> Option<F> {
        let mut total = F::zero();
        let mut count: usize = 0;

        for i in self {
            total = total + i.clone().into();
            count += 1;
        }

        if count == 0 {
            return None;
        }

        Some(total / F::from(count)?)
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
/// Returns `Ok(slope, intercept)` of the regression line.
pub fn lin_reg<'a, X, Y, IX, IY, F>(xs: IX, ys: IY, x_mean: F, y_mean: F) -> Result<(F, F), Error>
where
    X: 'a + Into<F> + Clone,
    Y: 'a + Into<F> + Clone,
    IX: Iterator<Item = &'a X>,
    IY: Iterator<Item = &'a Y>,
    F: FloatCore,
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
        return Err(Error::TooSteep);
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
/// * `xs` or `ys` do not have a mean (e.g. if they are empty, see `IteratorMean` for details)
/// * the slope is too steep to represent, approaching infinity
///
/// Returns `Ok(slope, intercept)` of the regression line.
pub fn linear_regression<X, Y, F>(xs: &[X], ys: &[Y]) -> Result<(F, F), Error>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: FloatCore,
{
    if xs.len() != ys.len() {
        return Err(Error::InputLenDif);
    }

    // if one of the axes is empty, we return `Error::DivByZero`
    let x_mean = xs.iter().mean().ok_or(Error::Mean)?;
    let y_mean = ys.iter().mean().ok_or(Error::Mean)?;

    lin_reg(xs.iter(), ys.iter(), x_mean, y_mean)
}

/// Linear regression from tuples.
///
/// Calculates the linear regression from a slice of tuple values.
///
/// Returns an error if
///
/// * `x` or `y` tuple members do not have a mean (e.g. if they are empty,
/// see [`IteratorMean`](./trait.IteratorMean.html) for details)
/// * the slope is too steep to represent, approaching infinity
///
/// Returns `Ok(slope, intercept)` of the regression line.
pub fn linear_regression_of<X, Y, F>(xys: &[(X, Y)]) -> Result<(F, F), Error>
where
    X: Clone + Into<F>,
    Y: Clone + Into<F>,
    F: FloatCore,
{
    let (count, x_total, y_total) =
        xys.iter()
            .fold((0, F::zero(), F::zero()), |acc: (usize, F, F), (x, y)| {
                (
                    acc.0 + 1,
                    acc.1 + x.clone().into(),
                    acc.2 + y.clone().into(),
                )
            });

    if count == 0 {
        return Err(Error::Mean);
    }

    let count = match F::from(count) {
        Some(f) => f,
        None => return Err(Error::Mean),
    };

    let x_mean = x_total / count;
    let y_mean = y_total / count;

    lin_reg(
        xys.iter().map(|(x, _)| x),
        xys.iter().map(|(_, y)| y),
        x_mean,
        y_mean,
    )
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;

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
