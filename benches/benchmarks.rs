// Combined benchmarks
use chrono::NaiveDate;
use criterion::{criterion_group, criterion_main, Criterion};

use rustframe::{
    frame::{Frame, RowIndex},
    matrix::{BoolMatrix, Matrix, SeriesOps},
    utils::{BDatesList, BDateFreq},
};
use std::time::Duration;

// Define size categories
const SIZES_SMALL: [usize; 1] = [1];
const SIZES_MEDIUM: [usize; 3] = [100, 250, 500];
const SIZES_LARGE: [usize; 1] = [1000];

// Modified benchmark functions to accept a slice of sizes
fn bool_matrix_operations_benchmark(c: &mut Criterion, sizes: &[usize]) {
    for &size in sizes {
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

fn matrix_boolean_operations_benchmark(c: &mut Criterion, sizes: &[usize]) {
    for &size in sizes {
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

fn matrix_operations_benchmark(c: &mut Criterion, sizes: &[usize]) {
    for &size in sizes {
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

    for &size in sizes {
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

fn generate_frame(size: usize) -> Frame<f64> {
    let data: Vec<f64> = (0..size * size).map(|x| x as f64).collect();
    let dates: Vec<NaiveDate> =
        BDatesList::from_n_periods("2000-01-01".to_string(), BDateFreq::Daily, size)
            .unwrap()
            .list()
            .unwrap();
    let col_names: Vec<String> = (1..=size).map(|i| format!("col_{}", i)).collect();
    Frame::new(
        Matrix::from_vec(data.clone(), size, size),
        col_names,
        Some(RowIndex::Date(dates)),
    )
}

fn benchmark_frame_operations(c: &mut Criterion, sizes: &[usize]) {
    for &size in sizes {
        let fa = generate_frame(size);
        let fb = generate_frame(size);

        c.bench_function(&format!("frame add ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &fa + &fb;
            });
        });

        c.bench_function(&format!("frame subtract ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &fa - &fb;
            });
        });

        c.bench_function(&format!("frame multiply ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &fa * &fb;
            });
        });

        c.bench_function(&format!("frame divide ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = &fa / &fb;
            });
        });

        c.bench_function(&format!("frame sum_horizontal ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = fa.sum_horizontal();
            });
        });
        c.bench_function(&format!("frame sum_vertical ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = fa.sum_vertical();
            });
        });
        c.bench_function(&format!("frame prod_horizontal ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = fa.prod_horizontal();
            });
        });
        c.bench_function(&format!("frame prod_vertical ({}x{})", size, size), |b| {
            b.iter(|| {
                let _result = fa.prod_vertical();
            });
        });
    }
}

// Runner functions for each size category
fn run_benchmarks_small(c: &mut Criterion) {
    bool_matrix_operations_benchmark(c, &SIZES_SMALL);
    matrix_boolean_operations_benchmark(c, &SIZES_SMALL);
    matrix_operations_benchmark(c, &SIZES_SMALL);
    benchmark_frame_operations(c, &SIZES_SMALL);
}

fn run_benchmarks_medium(c: &mut Criterion) {
    bool_matrix_operations_benchmark(c, &SIZES_MEDIUM);
    matrix_boolean_operations_benchmark(c, &SIZES_MEDIUM);
    matrix_operations_benchmark(c, &SIZES_MEDIUM);
    benchmark_frame_operations(c, &SIZES_MEDIUM);
}

fn run_benchmarks_large(c: &mut Criterion) {
    bool_matrix_operations_benchmark(c, &SIZES_LARGE);
    matrix_boolean_operations_benchmark(c, &SIZES_LARGE);
    matrix_operations_benchmark(c, &SIZES_LARGE);
    benchmark_frame_operations(c, &SIZES_LARGE);
}

// Configuration functions for different size categories
fn config_small_arrays() -> Criterion {
    Criterion::default()
        .sample_size(500)
        .measurement_time(Duration::from_millis(100))
        .warm_up_time(Duration::from_millis(5))
}

fn config_medium_arrays() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_millis(2000))
        .warm_up_time(Duration::from_millis(100))
}

fn config_large_arrays() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_millis(5000))
        .warm_up_time(Duration::from_millis(200))
}

criterion_group!(
    name = benches_small_arrays;
    config = config_small_arrays();
    targets = run_benchmarks_small
);
criterion_group!(
    name = benches_medium_arrays;
    config = config_medium_arrays();
    targets = run_benchmarks_medium
);
criterion_group!(
    name = benches_large_arrays;
    config = config_large_arrays();
    targets = run_benchmarks_large
);

criterion_main!(
    benches_small_arrays,
    benches_medium_arrays,
    benches_large_arrays
);
