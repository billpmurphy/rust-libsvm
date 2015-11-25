# rust-libsvm

[![Build Status](https://travis-ci.org/billpmurphy/rust-libsvm.svg)](https://travis-ci.org/billpmurphy/rust-libsvm)

Rust bindings and interface to [libsvm](https://github.com/cjlin1/libsvm).

**Status**: Unfinished


## Example usage

Build a model using dense data (with some custom parameters) and predict the
class of a new instance. Then, save the model.

```rust
extern crate libsvm;
use libsvm::*;

let y = vec![0.0, 1.0, 0.0];
let x = vec![vec![1.1, 0.0, 8.4],
             vec![0.9, 1.0, 9.1],
             vec![1.2, 1.0, 9.0]];

let problem = dense_problem(y, x).unwrap();

let mut params = SvmParam::new(2);
params.svm_type = SvmType::NuSVC;
params.kernel_type = KernelType::Sigmoid;
params.nu = 0.1;

let model = problem.train(&params).unwrap();

let x_new = vec![1.0, 0.0, 9.2];
let prediction = model.dense_predict(&x_new);
println!(prediction);

model.save(&"my_model");
```

Build a model using sparse data, scale it, and predict the class probabilites
of a new instance.

```rust
extern crate libsvm;
use libsvm::*;

let y = vec![0.0, 1.0, 0.0];
let x = vec![vec![(0, 1.1), (2, 8.4)],
             vec![(0, 0.9), (1, 1.0), (2, 9.1)],
             vec![(0, 1.2), (1, 1.0)];

let mut problem = dense_problem(y, x).unwrap();
problem.scale_x(Some(-1.0), Some(1.0));
problem.scale_y(Some(0.0), Some(1.0));

let mut params = SvmParam::new(2);
params.probability = true;

let model = problem.train(&params).unwrap();

let x_new = vec![(0, 1.0), (2, 9.2)];

let labels =
let probabilities = model.sparse_predict_probability(&x_new);
let class_probs = labels.iter().zip(model.get_labels().iter()).collect();
println!(class_probs);
```

Load a model from file and examine it. Then load a problem from file, and
perform cross-validation using the model.

```rust
extern crate libsvm;
use libsvm::*;

let modelfile = "tests/data/heart_scale.model";
let model = load_model(&modelfile).unwrap();

println!(model.get_num_sv()); // the number of support vectors
println!(model.get_svm_type());

let problem = load_problem(datafile).unwrap();

model.cross_validate(problem, 5); //5 folds
```
