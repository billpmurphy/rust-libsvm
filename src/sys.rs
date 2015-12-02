//! FFI bindings to the libsvm C++ library.

#[repr(C)]
pub struct CSVMProb {
    pub l: i32,
    pub y: *const f64,
    pub data: *const *const CSVMNode
}

#[repr(C)]
pub struct CSVMNode {
    pub index: i32,
    pub value: f64
}

#[allow(non_snake_case)]
#[repr(C)]
pub struct CSVMParameter {
    pub svm_type: i32,
    pub kernel_type: i32,
    pub degree: i32,
    pub gamma: f64,
    pub coef0: f64,
    pub cache_size: f64,
    pub eps: f64,
    pub C: f64,
    pub nr_weight: i32,
    pub weight_label: *const i32,
    pub weight: *const f64,
    pub nu: f64,
    pub p: f64,
    pub shrinking: i32,
    pub probability: i32
}

#[allow(non_snake_case)]
#[repr(C)]
pub struct CSVMModel {
    pub param: CSVMParameter,
    pub nr_class: i32,
    pub l: i32,
    pub svm_node: *mut *mut CSVMNode,
    pub sv_coef: *mut *mut f64,
    pub rho: *mut f64,
    pub probA: *mut f64,
    pub probB: *mut f64,
    pub sv_indices: *mut i32,
    pub label: *mut i32,
    pub nSV: *mut i32,
    pub free_sv: i32
}

extern "C" {
    pub static libsvm_version: i32;
    pub fn svm_save_model(model_file_name: *const i8, model: *const CSVMModel) -> i32;
    pub fn svm_load_model(model_file_name: *const i8) -> *mut CSVMModel;
    pub fn svm_train(prob: *const CSVMProb, param: *const CSVMParameter) -> *mut CSVMModel;
    pub fn svm_predict(model: *const CSVMModel, x: *const CSVMNode) -> f64;
    pub fn svm_cross_validation(prob: *const CSVMProb, param: *const CSVMParameter, nr_fold: i32,
                            target: *mut f64) -> ();
    pub fn svm_get_svm_type(model: *const CSVMModel) -> i32;
    pub fn svm_get_nr_class(model: *const CSVMModel) -> i32;
    pub fn svm_get_nr_sv(model: *const CSVMModel) -> i32;
    pub fn svm_get_labels(model: *const CSVMModel, label: *mut i32) -> ();
    pub fn svm_get_sv_indices(model: *const CSVMModel, sv_indices: *mut i32) -> ();
    pub fn svm_get_svr_probability(model: *const CSVMModel) -> f64;
    pub fn svm_predict_values(model: *const CSVMModel, x: *const CSVMNode,
                          dec_values: *const f64) -> f64;
    pub fn svm_predict_probability(model: *const CSVMModel, x: *const CSVMNode,
                               prob_estimates: *const f64) -> f64;
    pub fn svm_check_parameter(prob: *const CSVMProb, param: *const CSVMParameter) -> *const i8;
    pub fn svm_free_and_destroy_model(model_ptr_ptr: *const *const CSVMModel);
}
