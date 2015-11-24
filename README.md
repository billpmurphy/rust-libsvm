# rust-libsvm

Rust bindings and interface to [libsvm](https://github.com/cjlin1/libsvm).

**Status**: Unfinished


## Example usage

Build a model using dense data.

```rust
extern crate libsvm;
use libsvm::*;

let y = vec![0.0, 1.0, 0.0];
let x = vec![vec![1.1, 0.0, 8.4],
             vec![0.9, 1.0, 9.1],
             vec![1.2, 1.0, 9.0]];

let problem = dense_problem(y, x).unwrap();
let params = SvmParam::new(2);
let model = problem.train(&params).unwrap();

let x_new = vec![(0, 1.0), (2, 9.2)];
let prediction = model.predict(x_new);
```