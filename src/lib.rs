use std::iter::Iterator;

pub trait IteratorMean {
    fn mean(&mut self) -> Option<f64>;
}

impl<'a, T, I> IteratorMean for I
where
    T: 'a + Into<f64> + Clone,
    I: Iterator<Item = &'a T>,
{
    fn mean(&mut self) -> Option<f64> {
        let mut total = 0.0f64;
        let mut count: usize = 0;

        loop {
            if let Some(i) = self.next() {
                total += i.clone().into();
                count += 1;
            } else {
                break;
            }
        }

        if count > 0 {
            Some(total / count as f64)
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
    assert!(Vec::<u32>::new().iter().mean().is_none());
}

/// Linear regression
pub fn linear_regression<X, Y>(xs: &[X], ys: &[Y]) -> Option<(f64, f64)>
where
    X: Clone + Into<f64>,
    Y: Clone + Into<f64>,
{
    if xs.len() != ys.len() {
        return None;
    }

    // if one of the axes is empty, we return `None`
    let x_avg = xs.iter().mean()?;
    let y_avg = ys.iter().mean()?;

    // SUM (x-mean(x))^2
    let mut xxm2 = 0.0;

    // SUM (x-mean(x)) (y-mean(y))
    let mut xmym2 = 0.0;

    for (x, y) in xs.iter().zip(ys.iter()) {
        let x: f64 = x.clone().into();
        let y: f64 = y.clone().into();

        xxm2 += (x - x_avg).powi(2);
        xmym2 += (x - x_avg) * (y - y_avg);
    }

    let slope = xmym2 / xxm2;

    // we check for divide-by-zero after the fact
    if slope.is_nan() {
        return None;
    }

    let intercept = y_avg - slope * x_avg;

    Some((slope, intercept))
}

#[test]
fn test_example_regression() {
    let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];

    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
}

#[test]
fn test_integer_regression() {
    let xs: Vec<u8> = vec![1, 2, 3, 4, 5];
    let ys: Vec<u8> = vec![2, 4, 5, 4, 5];

    assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
}
