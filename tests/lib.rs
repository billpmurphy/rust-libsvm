use std::fs;

extern crate rust_libsvm;
use rust_libsvm::*;

#[test]
fn test_version() {
    assert_eq!(version(), 320);
}

#[test]
fn test_metrics() {
    let gt = vec![0, 0, 1, 0, 1, 2, 2, 3, 4, 4];
    let pr = vec![0, 0, 1, 0, 0, 2, 2, 4, 3, 3];

    assert_eq!(metrics::accuracy(&gt, &pr), 0.6);
    assert_eq!(metrics::class_recall(&gt, &pr, &0), 1.0);
    assert_eq!(metrics::class_recall(&gt, &pr, &1), 0.5);
    assert_eq!(metrics::class_recall(&gt, &pr, &2), 1.0);
    assert_eq!(metrics::class_recall(&gt, &pr, &3), 0.0);
    assert_eq!(metrics::class_recall(&gt, &pr, &4), 0.0);
    assert_eq!(metrics::class_precision(&gt, &pr, &0), 0.75);
    assert_eq!(metrics::class_precision(&gt, &pr, &1), 1.0);
    assert_eq!(metrics::class_precision(&gt, &pr, &2), 1.0);
    assert_eq!(metrics::class_precision(&gt, &pr, &3), 0.0);
    assert_eq!(metrics::class_precision(&gt, &pr, &4), 0.0);
    assert_eq!((metrics::class_f_measure(&gt, &pr, &0) * 100.0).round() as i32, 86);
    assert_eq!((metrics::class_f_measure(&gt, &pr, &1) * 100.0).round() as i32, 67);
    assert_eq!(metrics::class_f_measure(&gt, &pr, &2), 1.0);
    assert_eq!(metrics::class_f_measure(&gt, &pr, &3), 0.0);
    assert_eq!(metrics::class_f_measure(&gt, &pr, &4), 0.0);
}

#[test]
fn test_parse_data_from_file() {
    let datafile = "tests/data/heart_scale";
    check_data_file(datafile).unwrap();
    let problem = load_problem(datafile).unwrap();
    let mut params = SvmParam::new(2);
    params.probability = true;
    let model = problem.train(&params).unwrap();

    assert_eq!(model.get_num_classes(), 2);
}

#[test]
fn test_use_dense_data() {
    // Convert dense data, build RBF model
    let y = vec![0.0, 1.0, 0.0];
    let x = vec![vec![1.1, 0.0, 8.4],
                 vec![0.9, 1.0, 9.1],
                 vec![1.2, 1.0, 9.0]];

    let problem = dense_problem(y, x).unwrap();
    let params = SvmParam::new(2);
    let model = problem.train(&params).unwrap();

    assert_eq!(model.get_num_classes(), 2);
}

#[test]
fn test_use_sparse_data() {
    let y = vec![0.0, 1.0, 0.0];
    let x = vec![vec![(1, 0.1), (3, 0.2)],
                 vec![(3, 9.9)],
                 vec![(1, 0.2), (2, 3.2)]];
    let problem = sparse_problem(y, x).unwrap();
    let params = SvmParam::new(2);
    let model = problem.train(&params).unwrap();

    assert_eq!(model.get_num_classes(), 2);
}


#[test]
fn test_loaded_model() {
    // load the model
    let modelfile = "tests/data/heart_scale.model";
    let model = load_model(&modelfile).unwrap();

    // make sure the model is correct
    assert_eq!(model.get_num_classes(), 2); // check: num_classes
    assert_eq!(model.get_num_sv(), 91); // check: num sv
    assert_eq!(model.get_svr_probability(), 0.0); // check: svr prob (0, as this is classification)
    assert_eq!(model.get_svm_type(), SvmType::CSVC); // check: SvmType
    assert_eq!(model.get_num_sv(), 91); // check: number of support vectors
    assert_eq!(model.get_sv_indices().len(), 91); // check: support vector indices
    let labels = model.get_labels(); // check: get_labels
    assert_eq!(labels.len(), 2);
    assert!(labels.contains(&1));
    assert!(labels.contains(&-1));

    // predict some test instances
    let test_vec = vec![(1,0.708333), (2,1.0), (3,1.0), (4,-0.320755), (5,-0.105023), (6,-1.0),
                        (7,1.0), (8,-0.419847), (9,-1.0), (10,-0.225806), (12,1.0), (13,-1.0)];
    assert!([1.0, -1.0].contains(&model.sparse_predict(&test_vec)));

    // save and reload the model
    let new_modelfile = "tests/data/heart_scale_2.model";
    assert!(model.save(&new_modelfile).is_ok());
    assert!(load_model(&new_modelfile).is_ok());

    // clean up
    fs::remove_file(&new_modelfile).unwrap();
}
