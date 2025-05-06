// Combined benchmarks
use chrono::NaiveDate;
use criterion::{criterion_group, criterion_main, Criterion};

use rustframe::{
    frame::{Frame, RowIndex},
    matrix::{BoolMatrix, Matrix},
    utils::{BDateFreq, BDatesList},
};
use std::time::Duration;

pub fn for_short_runs() -> Criterion {
    Criterion::default()
        // (samples != total iterations)
        // limits the number of statistical data points.
        .sample_size(50)
        // measurement time per sample
        .measurement_time(Duration::from_millis(2000))
        // reduce warm-up time as well for faster overall run
        .warm_up_time(Duration::from_millis(50))
    // can make it much shorter if needed, e.g., 50ms measurement, 100ms warm-up
    // .measurement_time(Duration::from_millis(50))
    // .warm_up_time(Duration::from_millis(100))
}

const BENCH_SIZES: [usize; 5] = [1, 100, 250, 500, 1000];

fn bool_matrix_operations_benchmark(c: &mut Criterion) {
    let sizes = BENCH_SIZES;

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
    let sizes = BENCH_SIZES;

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
    let sizes = BENCH_SIZES;

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
    let n_periods = 1000;
    let n_cols = 1000;
    let dates: Vec<NaiveDate> =
        BDatesList::from_n_periods("2024-01-02".to_string(), BDateFreq::Daily, n_periods)
            .unwrap()
            .list()
            .unwrap();

    // let col_names= str(i) for i in range(1, 1000)
    let col_names: Vec<String> = (1..=n_cols).map(|i| format!("col_{}", i)).collect();

    let data1: Vec<f64> = (0..n_periods * n_cols).map(|x| x as f64).collect();
    let data2: Vec<f64> = (0..n_periods * n_cols).map(|x| (x + 1) as f64).collect();
    let ma = Matrix::from_vec(data1.clone(), n_periods, n_cols);
    let mb = Matrix::from_vec(data2.clone(), n_periods, n_cols);

    let fa = Frame::new(
        ma.clone(),
        col_names.clone(),
        Some(RowIndex::Date(dates.clone())),
    );
    let fb = Frame::new(mb, col_names, Some(RowIndex::Date(dates)));

    c.bench_function("frame element-wise multiply (1000x1000)", |b| {
        b.iter(|| {
            let _result = &fa * &fb;
        });
    });
}

// Define the criterion group and pass the custom configuration function
criterion_group!(
    name = combined_benches;
    config = for_short_runs(); // Use the custom configuration here
    targets = bool_matrix_operations_benchmark,
              matrix_boolean_operations_benchmark,
              matrix_operations_benchmark,
              benchmark_frame_operations
);
criterion_main!(combined_benches);
