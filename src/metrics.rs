//! Metrics for assessing SVM classification performance.

use std::f64;

// ============================================================================================= //
// Whole-dataset metrics

/// The squared correlation coefficient ( R^2 ). Regression problems only.
pub fn sq_correlation(ground_truth: &[f64], predicted: &[f64]) -> f64 {
    let pred_mean: f64 = predicted.iter().fold(0.0, |a, x| a + x) / predicted.len() as f64;
    let (res_sum_sq, tot_sum_sq) = ground_truth
        .iter()
        .zip(predicted.iter())
        .fold((0.0, 0.0), |(r, t), (gt, pr)| {
            (r + (gt - pr).powi(2), t + (gt - pred_mean).powi(2))
        });
    1.0 - (res_sum_sq / tot_sum_sq)
}

/// The weighted average accuracy of the prediction vector across all classes. Classification
/// problems only.
pub fn accuracy<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    let (correct, all) = ground_truth
        .iter()
        .zip(predicted.iter())
        .fold((0, 0), |(c, a), (gt, pr)| (c + (gt == pr) as i32, a + 1));
    (correct as f64) / (all as f64)
}

pub fn precision<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    unimplemented!()
}

pub fn recall<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    unimplemented!()
}

pub fn f_measure<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    unimplemented!()
}

pub fn roc_area<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    unimplemented!()
}

pub fn mean_sq_error<T: PartialEq>(ground_truth: &[T], predicted: &[T]) -> f64 {
    unimplemented!()
}

// ============================================================================================= //
// Class metrics

/// The recall of the predicted vector for the specified class.
pub fn class_recall<T: PartialEq>(ground_truth: &[T], predicted: &[T], class: &T) -> f64 {
    let (tp, tp_and_fn) =
        ground_truth
            .iter()
            .zip(predicted.iter())
            .fold((0, 0), |(t, g), (gt, pr)| {
                let class_t = gt == class;
                let correct = gt == pr;
                (t + (class_t && correct) as i32, g + class_t as i32)
            });
    (tp as f64) / (tp_and_fn) as f64
}

/// The precision of the predicted vector for the specified class.
pub fn class_precision<T: PartialEq>(ground_truth: &[T], predicted: &[T], class: &T) -> f64 {
    let (tp, tp_and_fp) =
        ground_truth
            .iter()
            .zip(predicted.iter())
            .fold((0, 0), |(t, g), (gt, pr)| {
                let pred_t = pr == class;
                let correct = gt == pr;
                (t + (pred_t && correct) as i32, g + pred_t as i32)
            });
    (tp as f64) / (tp_and_fp) as f64
}

/// The F1 Score (F-Measure) of the predicted vector for the specified class.
pub fn class_f_measure<T: PartialEq>(ground_truth: &[T], predicted: &[T], class: &T) -> f64 {
    let (tp, fn_and_fp) =
        ground_truth
            .iter()
            .zip(predicted.iter())
            .fold((0, 0), |(t, g), (gt, pr)| {
                let class_t = gt == class;
                let pred_t = pr == class;
                let correct = gt == pr;
                (
                    t + (pred_t && correct) as i32,
                    g + ((class_t || pred_t) && !correct) as i32,
                )
            });
    (tp as f64) * 2.0 / ((tp as f64 * 2.0) + fn_and_fp as f64)
}

pub fn class_roc_area<T: PartialEq>(ground_truth: &[T], predicted: &[T], class: &T) -> f64 {
    unimplemented!()
}

pub fn class_mean_sq_error<T: PartialEq>(ground_truth: &[T], predicted: &[T], class: &T) -> f64 {
    unimplemented!()
}
