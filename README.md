linreg-rs
=========

`linreg` is a small crate that calculates linear regressions. It works without
stdlib or memory allocation and has few dependencies. Example:

```rust
let xs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let ys: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];

assert_eq!(Some((0.6, 2.2)), linear_regression(&xs, &ys));
```

It supports tubles, separate vectors for x and y values and template floating point types.
