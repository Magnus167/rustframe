// Combined benchmarks for rustframe
use chrono::NaiveDate;
use criterion::{criterion_group, criterion_main, Criterion};
use rustframe::{
    frame::{Frame, RowIndex},
    matrix::{BoolMatrix, Matrix},
    utils::{BDateFreq, BDatesList},
};

fn bool_matrix_operations_benchmark(c: &mut Criterion) {
    // let sizes = [1, 100, 1000];
    let sizes = [1000];

    for &size in &sizes {
        let data1: Vec<bool> = (0..size * size).map(|x| x % 2 == 0).collect();
        let data2: Vec<bool> = (0..size * size).map(|x| x % 3 == 0).collect();
        let bm1 = BoolMatrix::from_vec(data1.clone(), size, size);
        let bm2 = BoolMatrix::from_vec(data2.clone(), size, size);

        c.bench_function(&format!("bool_matrix_and ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 & &bm2;
            });
        });

        c.bench_function(&format!("bool_matrix_or ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 | &bm2;
            });
        });

        c.bench_function(&format!("bool_matrix_xor ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 ^ &bm2;
            });
        });

        c.bench_function(&format!("bool_matrix_not ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = !&bm1;
            });
        });
    }
}

fn matrix_boolean_operations_benchmark(c: &mut Criterion) {
    // let sizes = [1, 100, 1000];
    let sizes = [1000];

    for &size in &sizes {
        let data1: Vec<bool> = (0..size * size).map(|x| x % 2 == 0).collect();
        let data2: Vec<bool> = (0..size * size).map(|x| x % 3 == 0).collect();
        let bm1 = BoolMatrix::from_vec(data1.clone(), size, size);
        let bm2 = BoolMatrix::from_vec(data2.clone(), size, size);

        c.bench_function(&format!("boolean AND ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 & &bm2;
            });
        });

        c.bench_function(&format!("boolean OR ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 | &bm2;
            });
        });

        c.bench_function(&format!("boolean XOR ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &bm1 ^ &bm2;
            });
        });

        c.bench_function(&format!("boolean NOT ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = !&bm1;
            });
        });
    }
}

fn matrix_operations_benchmark(c: &mut Criterion) {
    // let sizes = [1, 100, 1000];
    let sizes = [1000];

    for &size in &sizes {
        let data: Vec<f64> = (0..size * size).map(|x| x as f64).collect();
        let ma = Matrix::from_vec(data.clone(), size, size);

        c.bench_function(&format!("scalar add ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma + 1.0;
            });
        });

        c.bench_function(&format!("scalar subtract ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma - 1.0;
            });
        });

        c.bench_function(&format!("scalar multiply ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma * 2.0;
            });
        });

        c.bench_function(&format!("scalar divide ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma / 2.0;
            });
        });
    }

    // Benchmarking matrix addition
    for &size in &sizes {
        let data1: Vec<f64> = (0..size * size).map(|x| x as f64).collect();
        let data2: Vec<f64> = (0..size * size).map(|x| (x + 1) as f64).collect();
        let ma = Matrix::from_vec(data1.clone(), size, size);
        let mb = Matrix::from_vec(data2.clone(), size, size);

        c.bench_function(&format!("matrix add ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma + &mb;
            });
        });

        c.bench_function(&format!("matrix subtract ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma - &mb;
            });
        });

        c.bench_function(&format!("matrix multiply ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma * &mb;
            });
        });

        c.bench_function(&format!("matrix divide ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &ma / &mb;
            });
        });
    }
}

fn benchmark_frame_operations(c: &mut Criterion) {
    let n_periods = 4;
    let dates: Vec<NaiveDate> =
        BDatesList::from_n_periods("2024-01-02".to_string(), BDateFreq::Daily, n_periods)
            .unwrap()
            .list()
            .unwrap();

    let col_names: Vec<String> = vec!["a".to_string(), "b".to_string()];

    let ma = Matrix::from_cols(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
    let mb = Matrix::from_cols(vec![vec![4.0, 3.0, 2.0, 1.0], vec![8.0, 7.0, 6.0, 5.0]]);

    let fa = Frame::new(
        ma.clone(),
        col_names.clone(),
        Some(RowIndex::Date(dates.clone())),
    );
    let fb = Frame::new(mb, col_names, Some(RowIndex::Date(dates)));

    c.bench_function("frame element-wise multiply", |b| {
        b.iter(|| {
            let _result = &fa * &fb;
        });
    });
}

criterion_group!(
    combined_benches,
    bool_matrix_operations_benchmark,
    matrix_boolean_operations_benchmark,
    matrix_operations_benchmark,
    benchmark_frame_operations
);
criterion_main!(combined_benches);
