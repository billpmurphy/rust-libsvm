/// Rust API and wrapper to the popular [libsvm](https://github.com/cjlin1/libsvm) library.

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::io::{BufReader, BufRead};
use std::str;

pub mod metrics;

// ============================================================================================= //
// FFI bindings to libsvm

#[repr(C)]
struct CSVMProb {
    l: i32,
    y: *const f64,
    data: *const *const CSVMNode
}

#[repr(C)]
struct CSVMNode {
    index: i32,
    value: f64
}

#[allow(non_snake_case)]
#[repr(C)]
struct CSVMParameter {
    svm_type: i32,
    kernel_type: i32,
    degree: i32,
    gamma: f64,
    coef0: f64,
    cache_size: f64,
    eps: f64,
    C: f64,
    nr_weight: i32,
    weight_label: *const i32,
    weight: *const f64,
    nu: f64,
    p: f64,
    shrinking: i32,
    probability: i32
}

#[allow(non_snake_case)]
#[repr(C)]
struct CSVMModel {
    param: CSVMParameter,
    nr_class: i32,
    l: i32,
    svm_node: *mut *mut CSVMNode,
    sv_coef: *mut *mut f64,
    rho: *mut f64,
    probA: *mut f64,
    probB: *mut f64,
    sv_indices: *mut i32,
    label: *mut i32,
    nSV: *mut i32,
    free_sv: i32
}

extern "C" {
    static libsvm_version: i32;
    fn svm_save_model(model_file_name: *const i8, model: *const CSVMModel) -> i32;
    fn svm_load_model(model_file_name: *const i8) -> *mut CSVMModel;
    fn svm_train(prob: *const CSVMProb, param: *const CSVMParameter) -> *mut CSVMModel;
    fn svm_predict(model: *const CSVMModel, x: *const CSVMNode) -> f64;
    fn svm_cross_validation(prob: *const CSVMProb, param: *const CSVMParameter, nr_fold: i32,
                            target: *mut f64) -> ();
    fn svm_get_svm_type(model: *const CSVMModel) -> i32;
    fn svm_get_nr_class(model: *const CSVMModel) -> i32;
    fn svm_get_nr_sv(model: *const CSVMModel) -> i32;
    fn svm_get_labels(model: *const CSVMModel, label: *mut i32) -> ();
    fn svm_get_sv_indices(model: *const CSVMModel, sv_indices: *mut i32) -> ();
    fn svm_get_svr_probability(model: *const CSVMModel) -> f64;
    fn svm_predict_values(model: *const CSVMModel, x: *const CSVMNode,
                          dec_values: *const f64) -> f64;
    fn svm_predict_probability(model: *const CSVMModel, x: *const CSVMNode,
                               prob_estimates: *const f64) -> f64;
    fn svm_check_parameter(prob: *const CSVMProb, param: *const CSVMParameter) -> *const i8;
    fn svm_free_and_destroy_model(model_ptr_ptr: *const *const CSVMModel);
}

// ============================================================================================= //
// Rust API

type SvmNode = CSVMNode;

/// Error type for Rust libsvm API.
#[derive(Debug)]
pub enum SvmErr {
    /// Invalid parameter values in an SvmParam object.
    ParamError(String),
    /// Input data is incorrectly structured.
    InvalidDataError(String),
    /// I/O error reading/writing data to file.
    IoError(io::Error),
    /// Error parsing data from file; data is incorrectly formatted.
    ParseError(String),
    /// Error loading/saving SvmModel to/from file.
    ModelError(String),
}

/// The type of Support Vector Machine used in a model.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SvmType {
    /// C-SVM classification
    CSVC = 0,
    /// nu-SVM classification
    NuSVC = 1,
    /// one-class-SVM
    OneClass = 2,
    /// epsilon-SVM regression
    EpsilonSVR = 3,
    /// nu-SVM regression
    NuSVR = 4
}

/// The type of the kernel function.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KernelType {
    /// u'*v
    Linear = 0,
    /// (gamma*u'*v + coef0)^degree
    Poly = 1,
    /// exp(-gamma*|u-v|^2)
    RBF = 2,
    /// tanh(gamma*u'*v + coef0)
    Sigmoid = 3,
    /// kernel values pre-set
    Precomputed = 4
}

/// Return the current version number of libsvm.
pub fn version() -> i32 {
    libsvm_version
}

///
pub struct SvmModel {
    c_model: *const CSVMModel
}

impl Drop for SvmModel {
    fn drop(&mut self) {
        let model_ptr_ptr: *const *const CSVMModel = &self.c_model;
        unsafe { svm_free_and_destroy_model(model_ptr_ptr) }
    }
}

impl SvmModel {
    /// Save model to file.
    pub fn save(&self, filename: &str) -> Result<(), SvmErr> {
        let m = &self.c_model;
        let cfilename = try!(CString::new(filename)
                            .map_err(|_| SvmErr::ModelError("Invalid filename.".to_owned())));
        unsafe {
            match svm_save_model(cfilename.as_ptr(), *m) {
                0 => Ok(()),
                n => Err(SvmErr::ModelError(format!("Failed to save model. Error code: {}", n)))
            }
        }
    }

    /// Return the SvmType of the model.
    pub fn get_svm_type(&self) -> SvmType {
        unsafe { std::mem::transmute(svm_get_svm_type(self.c_model) as i8) }
    }

    /// For a classification model, return the number of classes. For a regression or a one-class
    /// model, return 2.
    pub fn get_num_classes(&self) -> i32 {
        unsafe { svm_get_nr_class(self.c_model) }
    }

    /// Return the number of total support vectors.
    pub fn get_num_sv(&self) -> i32 {
        unsafe { svm_get_nr_sv(self.c_model) }
    }

    /// Return a vector of class labels.
    pub fn get_labels(&self) -> Vec<i32> {
        let num_labels = self.get_num_classes() as usize;
        let mut labels = vec![0; num_labels];
        unsafe { svm_get_labels(self.c_model, labels.as_mut_ptr()); }
        labels
    }

    /// For a regression model with probability information, this output a value sigma > 0.
    /// For test data, we consider the probability model: target value = predicted value + z, z:
    /// Laplace distribution e^(-|z|/sigma)/(2sigma). If the model is not for svr or does not
    /// contain required information, return 0.
    pub fn get_svr_probability(&self) -> f64 {
        unsafe { svm_get_svr_probability(self.c_model) }
    }

    /// Return the indicies of the support vectors.
    pub fn get_sv_indices(&self) -> Vec<i32> {
        let num_svs = self.get_num_sv() as usize;
        let mut indicies = vec![0; num_svs];
        unsafe { svm_get_sv_indices(self.c_model, indicies.as_mut_ptr()); }
        indicies
    }

    /// Do classification or regression on a test sparse vector x given a model.
    ///
    /// For a classification model, the predicted class for x is returned. For a regression model,
    /// the function value of x calculated using the model is returned. For an one-class model, +1
    /// or -1 is returned.
    pub fn sparse_predict(&self, feature_vec: &[(i32, f64)]) -> f64 {
        let svm_nodes: Vec<SvmNode> = feature_vec
            .iter()
            .map(|&(i, c)| SvmNode { index: i, value: c })
            .collect();
        let c_model: *const CSVMModel = self.c_model;
        let c_vec: *const SvmNode = svm_nodes.as_ptr();
        unsafe {
            svm_predict(c_model, c_vec) as f64
        }
    }

    /// Do classification or regression on a test dense vector x given a model.
    ///
    /// For a classification model, the predicted class for x is returned. For a regression model,
    /// the function value of x calculated using the model is returned. For an one-class model, +1
    /// or -1 is returned.
    pub fn dense_predict(&self, feature_vec: &[f64]) -> f64 {
        let sparse_vec: Vec<(i32, f64)> = feature_vec.iter()
            .filter(|x| **x != 0.0)
            .enumerate()
            .map(|(i, &x)| (i as i32, x))
            .collect();
        self.sparse_predict(&sparse_vec)
    }

    /// Do classification or regression on a sparse test vector x given a model with probability
    /// information.
    pub fn sparse_predict_probability(&self, feature_vec: &[(i32, f64)]) -> Vec<f64>{
        panic!("Not implemented yet.")
    }

    /// Do classification or regression on a dense test vector x given a model with probability
    /// information.
    pub fn dense_predict_probability(&self, feature_vec: &[f64]) -> Vec<f64>{
        panic!("Not implemented yet.")
    }

    /// Conduct cross-validation. Data are separated to num_folds folds. Under given parameters,
    /// sequentially each fold is validated using the model from training the remaining.
    pub fn cross_validation(&self, prob: &SvmProb, num_folds: i32) -> Vec<f64> {
        panic!("Not implemented yet")
    }
}

/// The parameters of an SvmModel.
pub struct SvmParam {
    /// The type of SVM.
    svm_type: SvmType,
    /// The type of kernel function.
    kernel_type: KernelType,
    /// The degree of the kernel function.
    degree: i32,
    /// The gamma in the kernel function.
    gamma: f64,
    /// The coef0 in the kernel function.
    coef0: f64,
    /// The cache size in MB.
    cache_size: f64,
    /// Tolerance of stopping criterion.
    eps: f64,
    /// Cost of constraints violation for CSVC.
    c: f64,
    /// Vector of weight labels for each class.
    weight_label: Vec<i32>,
    /// The penalty of each class is scaled by the factor in this vector.
    weight: Vec<f64>,
    /// Parameter nu of nu-SVC, one-class SVM, and nu-SVR.
    nu: f64,
    /// Epsilon in loss function of epsilon-SVR.
    p: f64,
    /// Whether to use shrinking heuristics in the model.
    shrinking: bool,
    /// Whether to train a SVC or SVR model for probability estimates.
    probability: bool
}


impl SvmParam {
    /// Create a new instance of SvmParam given the number of classes in the dataset, All of the
    /// parameter values will be set to the libsvm defaults.
    pub fn new(classes: usize) -> SvmParam {
        SvmParam {
            svm_type: SvmType::CSVC,
            kernel_type: KernelType::RBF,
            degree: 3,
            gamma: 1.0/classes as f64,
            coef0: 0.0,
            cache_size: 100.0,
            eps: 0.001,
            c: 1.0,
            weight_label: vec![1; classes],
            weight: vec![1.0; classes],
            nu: 0.5,
            p: 0.1,
            shrinking: true,
            probability: false
        }
    }

    fn to_c(&self) -> CSVMParameter {
        CSVMParameter {
            svm_type: self.svm_type.clone() as i32,
            kernel_type: self.kernel_type.clone() as i32,
            degree: self.degree.clone(),
            gamma: self.gamma.clone(),
            coef0: self.coef0.clone(),
            cache_size: self.cache_size,
            eps: self.eps.clone(),
            C: self.c.clone(),
            nr_weight: self.weight_label.len() as i32,
            weight_label: self.weight_label.clone().as_ptr(),
            weight: self.weight.clone().as_ptr(),
            nu: self.nu.clone(),
            p: self.p.clone(),
            shrinking: self.shrinking.clone() as i32,
            probability: self.probability.clone() as i32
        }
    }
}

/// Describes a dataset, consisting of an output vector and a series of feature vectors in sparse
/// format.
pub struct SvmProb {
    c_problem: CSVMProb
}


impl SvmProb {
    /// Save problem to file.
    pub fn save(&self, filename: &str) -> Result<(), SvmErr> {
        panic!("Not implemented yet.")
    }

    /// Construct and return an SVM model according to the given training data and parameters.
    pub fn train(&self, parameters: &SvmParam) -> Result<SvmModel, SvmErr> {
        let c_prob = &self.c_problem;
        let c_params: *const CSVMParameter = &parameters.to_c();
        unsafe {
            let result: *const i8 = svm_check_parameter(c_prob, c_params);
            if result.is_null() {
                Ok(SvmModel { c_model: svm_train(c_prob, c_params) })
            } else {
                let c_str: &CStr = CStr::from_ptr(result);
                let buf: &[u8] = c_str.to_bytes();
                Err(SvmErr::ParamError(match str::from_utf8(buf) {
                    Ok(s) => format!("Invalid parameters. Error message: {}", s),
                    Err(_) => "Invalid parameters.".to_owned()
                }))
            }
        }
    }

    /// Scale the feature vectors to [lower, upper].
    pub fn scale_x(&mut self, lower: &Option<f64>, upper: &Option<f64>) {
        panic!("Not implemented yet.")
    }

    /// Scale the y vector to [lower, upper].
    pub fn scale_y(&mut self, lower: &Option<f64>, upper: &Option<f64>) {
        panic!("Not implemented yet.")
    }
}


/// Create an SvmProb from a vector of output variables and a vector of feature vectors represented
/// as (index, value) pairs.
pub fn sparse_problem(y: Vec<f64>, x: Vec<Vec<(i32, f64)>>) -> Result<SvmProb, SvmErr> {
    // Data check: Problem is not empty
    if y.len() == 0 || x.len() == 0 {
        return Err(SvmErr::InvalidDataError(format!("Cannot create problem from empty vector")))
    }

    // Data check: output and input vectors have the same length
    let len = y.len();
    if x.len() != len {
        return Err(SvmErr::InvalidDataError(
            format!("y vector has length {} but x vector has length {}", len, x.len())))
    }

    // Build the problem struct
    let mut x_data = Vec::new();
    for row in x {
        let mut new_row = Vec::new();
        for (i, cell) in row {
            if cell != 0.0 {
                new_row.push(CSVMNode { index: i, value: cell });
            }
        }
        new_row.push(CSVMNode { index: -1, value: 0.0 });
        x_data.push(new_row.as_ptr());
    }

    Ok(SvmProb {
        c_problem: CSVMProb {
            l: len as i32,
            y: y.as_ptr(),
            data: x_data.as_ptr()
        }
    })
}


/// Create a SvmProb from a one-dimensional output vector and a two-dimensional input vector.
pub fn dense_problem(y: Vec<f64>, x: Vec<Vec<f64>>) -> Result<SvmProb, SvmErr> {
    // Data check: Problem is not empty
    if y.len() == 0 || x.len() == 0 {
        return Err(SvmErr::InvalidDataError(format!("Cannot create problem from empty vector")))
    }

    // Check data
    let len = y.len();
    if x.len() != len {
        return Err(SvmErr::InvalidDataError(
            format!("y vector has length {} but x vector has length {}", len, x.len())))
    }

    // Convert the data into a vector of C arrays of SVMNode structs
    let mut x_data: Vec<*const CSVMNode> = Vec::new();
    for row in x {
        let mut new_row = Vec::new();
        for (i, &cell) in row.iter().enumerate() {
            if i != 0 {
                new_row.push(CSVMNode { index: i as i32, value: cell });
            }
        }
        new_row.push(CSVMNode { index: -1, value: 0.0 });
        x_data.push(new_row.as_ptr());
    }

    // Build the SvmProb struct
    Ok(SvmProb {
        c_problem: CSVMProb {
            l: len as i32,
            y: y.as_ptr(),
            data: x_data.as_ptr()
        }
    })
}

/// Load a model from file.
pub fn load_model(filename: &str) -> Result<SvmModel, SvmErr> {
    let cfilename = try!(CString::new(filename)
                         .map_err(|_| SvmErr::ModelError("Invalid filename.".to_owned())));
    unsafe {
        let model: *const CSVMModel = svm_load_model(cfilename.as_ptr());
        if model.is_null() {
            Err(SvmErr::ModelError("Failed to load model.".to_owned()))
        } else {
            Ok(SvmModel { c_model: model })
        }
    }
}


/// Load a problem from file.
pub fn load_problem(filename: &str) -> Result<SvmProb, SvmErr> {
    let f = try!(File::open(filename).map_err(SvmErr::IoError));
    let file = BufReader::new(&f);

    let mut y = Vec::new();
    let mut x = Vec::new();
    for line in file.lines() {
        let line_str = try!(line.map_err(SvmErr::IoError));
        let (y_line, x_line) = try!(parse_line(&line_str));
        y.push(y_line);
        x.push(x_line);
    }
    sparse_problem(y, x)
}

/// Check whether a file is in the correct data format.
pub fn check_data_file(filename: &str) -> Result<(), SvmErr> {
    let f = try!(File::open(filename).map_err(SvmErr::IoError));
    let file = BufReader::new(&f);
    for line in file.lines() {
        let line_str = try!(line.map_err(SvmErr::IoError));
        try!(parse_line(&line_str));
    }
    Ok(())
}


// ============================================================================================= //
// libsvm file format parsing utilities

/// Parse a line from the svm file format. A line consists of an output variable and a list of
/// <index>:value pairs, separated by spaces. The line must end with an <index>:<value> pair with
/// an index of -1.
///
/// Examples:
/// +1 1:1, 2:-2.3 -1:0
/// -1
/// 2
fn parse_line(linestr: &str) -> Result<(f64, Vec<(i32, f64)>), SvmErr> {
    let line = linestr.trim();
    let tokens: Vec<&str> = line.split(' ').collect();
    match tokens.len() {
        0 => Err(SvmErr::ParseError("Failed to parse empty line".to_owned())),
        1 => Ok((try!(parse_output_var(tokens[0])), Vec::new())),
        _ => {
            let y = try!(parse_output_var(tokens[0]));
            let mut xs = Vec::new();
            for token in &tokens[1..] {
                let line_pairs = try!(parse_index_value_pair(token));
                xs.push(line_pairs);
            }
            Ok((y, xs))
        }
    }
}


/// Parse an <index>:<value> pair from the svm file format. An <index>:<value> pair consists of an
/// i32-valued index and an f64-valued cell value, separated by a colon.
/// Examples: 1:1, 2:2, 3:3.0, 4:-4, 5:-5.5
fn parse_index_value_pair(s: &str) -> Result<(i32, f64), SvmErr> {
    let tokens: Vec<&str> = s.split(':').collect();
    if tokens.len() == 2 {
        let index = try!(tokens[0].parse().map_err(|_|
                        SvmErr::ParseError(format!("Failed to parse index {}",
                                                  tokens[0]))));
        let value = try!(tokens[1].parse().map_err(|_|
                        SvmErr::ParseError(format!("Failed to parse input variable {}",
                                                  tokens[1]))));
        Ok((index, value))
    } else {
        Err(SvmErr::ParseError(format!("Expected <index>:<value> pair, found {}", s)))
    }
}

/// Parse an output variable from the svm file format. An output variable is an f64 value.
/// Examples: +1, 2, -3, +4.0, 5.55, -6.0
fn parse_output_var(s: &str) -> Result<f64, SvmErr> {
    let mut chars = s.chars();
    match chars.next() {
        Some('+') => parse_output_var(&chars.collect::<String>()),
        Some(_) => s.parse().map_err(|_|
            SvmErr::ParseError(format!("Failed to parse output variable {}", s))),
        None => Err(SvmErr::ParseError("Expected output variable, found nothing".to_owned()))
    }
}
