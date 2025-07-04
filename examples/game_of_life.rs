// src/gol.rs

use rand::{self, Rng};
use rustframe::matrix::{BoolMatrix, IntMatrix, Matrix};

/// Helper function to create a shifted version of the game board.
/// (Using the version provided by the user)
///
/// - `game`: The current state of the Game of Life as a `BoolMatrix`.
/// - `dr`: The row shift (delta row). Positive shifts down, negative shifts up.
/// - `dc`: The column shift (delta column). Positive shifts right, negative shifts left.
///
/// Returns an `IntMatrix` of the same dimensions as `game`.
/// - Cells in the shifted matrix get value `1` if the corresponding source cell in `game` was `true` (alive).
/// - Cells that would source from outside `game`'s bounds (due to the shift) get value `0`.
fn get_shifted_neighbor_layer(game: &BoolMatrix, dr: isize, dc: isize) -> IntMatrix {
    let rows = game.rows();
    let cols = game.cols();

    if rows == 0 || cols == 0 {
        // Handle 0x0 case, other 0-dim cases panic in Matrix::from_vec
        return IntMatrix::from_vec(vec![], 0, 0);
    }

    // Initialize with a matrix of 0s using from_vec.
    // This demonstrates creating an IntMatrix and then populating it.
    let mut shifted_layer = IntMatrix::from_vec(vec![0i32; rows * cols], rows, cols);

    for r_target in 0..rows {
        // Iterate over cells in the *new* (target) shifted matrix
        for c_target in 0..cols {
            // Calculate where this target cell would have come from in the *original* game matrix
            let r_source = r_target as isize - dr;
            let c_source = c_target as isize - dc;

            // Check if the source coordinates are within the bounds of the original game matrix
            if r_source >= 0
                && r_source < rows as isize
                && c_source >= 0
                && c_source < cols as isize
            {
                // If the source cell in the original game was alive...
                if game[(r_source as usize, c_source as usize)] {
                    // Demonstrates Index access on BoolMatrix
                    // ...then this cell in the shifted layer is 1.
                    shifted_layer[(r_target, c_target)] = 1; // Demonstrates IndexMut access on IntMatrix
                }
            }
            // Else (source is out of bounds): it remains 0, as initialized.
        }
    }
    shifted_layer // Return the constructed IntMatrix
}

/// Calculates the next generation of Conway's Game of Life.
///
/// This implementation uses a broadcast-like approach by creating shifted layers
/// for each neighbor and summing them up, then applying rules element-wise.
///
/// - `current_game`: A `&BoolMatrix` representing the current state (true=alive).
///
/// Returns: A new `BoolMatrix` for the next generation.
pub fn game_of_life_next_frame(current_game: &BoolMatrix) -> BoolMatrix {
    let rows = current_game.rows();
    let cols = current_game.cols();

    if rows == 0 && cols == 0 {
        return BoolMatrix::from_vec(vec![], 0, 0); // Return an empty BoolMatrix
    }
    // Assuming valid non-empty dimensions (e.g., 25x25) as per typical GOL.
    // Your Matrix::from_vec would panic for other invalid 0-dim cases.

    // Define the 8 neighbor offsets (row_delta, col_delta)
    let neighbor_offsets: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1), // Top row (NW, N, NE)
        (0, -1),
        (0, 1), // Middle row (W, E)
        (1, -1),
        (1, 0),
        (1, 1), // Bottom row (SW, S, SE)
    ];

    // 1. Initialize `neighbor_counts` with the first shifted layer.
    //    This demonstrates creating an IntMatrix from a function and using it as a base.
    let (first_dr, first_dc) = neighbor_offsets[0];
    let mut neighbor_counts = get_shifted_neighbor_layer(current_game, first_dr, first_dc);

    // 2. Add the remaining 7 neighbor layers.
    //    This demonstrates element-wise addition of matrices (`Matrix + Matrix`).
    for i in 1..neighbor_offsets.len() {
        let (dr, dc) = neighbor_offsets[i];
        let next_neighbor_layer = get_shifted_neighbor_layer(current_game, dr, dc);
        // `neighbor_counts` (owned IntMatrix) + `next_neighbor_layer` (owned IntMatrix)
        // uses `impl Add for Matrix`, consumes both, returns new owned `IntMatrix`.
        neighbor_counts = neighbor_counts + next_neighbor_layer;
    }

    // 3. Apply Game of Life rules using element-wise operations.

    // Rule: Survival or Birth based on neighbor counts.
    // A cell is alive in the next generation if:
    //   (it's currently alive AND has 2 or 3 neighbors) OR
    //   (it's currently dead AND has exactly 3 neighbors)

    // `neighbor_counts.eq_elem(scalar)`:
    //   Demonstrates element-wise comparison of a Matrix with a scalar (broadcast).
    //   Returns an owned `BoolMatrix`.
    let has_2_neighbors = neighbor_counts.eq_elem(2);
    let has_3_neighbors = neighbor_counts.eq_elem(3); // This will be reused

    // `has_2_neighbors | has_3_neighbors`:
    //   Demonstrates element-wise OR (`Matrix<bool> | Matrix<bool>`).
    //   Consumes both operands, returns an owned `BoolMatrix`.
    let has_2_or_3_neighbors = has_2_neighbors | has_3_neighbors.clone(); // Clone has_3_neighbors as it's used again

    // `current_game & &has_2_or_3_neighbors`:
    //   `current_game` is `&BoolMatrix`. `has_2_or_3_neighbors` is owned.
    //   Demonstrates element-wise AND (`&Matrix<bool> & &Matrix<bool>`).
    //   Borrows both operands, returns an owned `BoolMatrix`.
    let survives = current_game & &has_2_or_3_neighbors;

    // `!current_game`:
    //   Demonstrates element-wise NOT (`!&Matrix<bool>`).
    //   Borrows operand, returns an owned `BoolMatrix`.
    let is_dead = !current_game;

    // `is_dead & &has_3_neighbors`:
    //   `is_dead` is owned. `has_3_neighbors` is owned.
    //   Demonstrates element-wise AND (`Matrix<bool> & &Matrix<bool>`).
    //   Consumes `is_dead`, borrows `has_3_neighbors`, returns an owned `BoolMatrix`.
    let births = is_dead & &has_3_neighbors;

    // `survives | births`:
    //   Demonstrates element-wise OR (`Matrix<bool> | Matrix<bool>`).
    //   Consumes both operands, returns an owned `BoolMatrix`.
    let next_frame_game = survives | births;

    next_frame_game
}

pub fn generate_glider(board: &mut BoolMatrix, board_size: usize) {
    // Initialize with a Glider pattern.
    // It demonstrates how to set specific cells in the matrix.
    // This demonstrates `IndexMut` for `current_board[(r, c)] = true;`.
    let mut rng = rand::rng();
    let r_offset = rng.random_range(0..(board_size - 3));
    let c_offset = rng.random_range(0..(board_size - 3));
    if board.rows() >= r_offset + 3 && board.cols() >= c_offset + 3 {
        board[(r_offset + 0, c_offset + 1)] = true;
        board[(r_offset + 1, c_offset + 2)] = true;
        board[(r_offset + 2, c_offset + 0)] = true;
        board[(r_offset + 2, c_offset + 1)] = true;
        board[(r_offset + 2, c_offset + 2)] = true;
    }
}

pub fn generate_pulsar(board: &mut BoolMatrix, board_size: usize) {
    // Initialize with a Pulsar pattern.
    // This demonstrates how to set specific cells in the matrix.
    // This demonstrates `IndexMut` for `current_board[(r, c)] = true;`.
    let mut rng = rand::rng();
    let r_offset = rng.random_range(0..(board_size - 17));
    let c_offset = rng.random_range(0..(board_size - 17));
    if board.rows() >= r_offset + 17 && board.cols() >= c_offset + 17 {
        let pulsar_coords = [
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 10),
            (2, 11),
            (2, 12),
            (4, 2),
            (4, 7),
            (4, 9),
            (4, 14),
            (5, 2),
            (5, 7),
            (5, 9),
            (5, 14),
            (6, 2),
            (6, 7),
            (6, 9),
            (6, 14),
            (7, 4),
            (7, 5),
            (7, 6),
            (7, 10),
            (7, 11),
            (7, 12),
        ];
        for &(dr, dc) in pulsar_coords.iter() {
            board[(r_offset + dr, c_offset + dc)] = true;
        }
    }
}

pub fn detect_stable_state(
    current_board: &BoolMatrix,
    previous_board_state: &Option<BoolMatrix>,
) -> bool {
    if let Some(ref prev_board) = previous_board_state {
        // `*prev_board == current_board` demonstrates `PartialEq` for `Matrix`.
        return *prev_board == *current_board;
    }
    false
}

pub fn hash_board(board: &BoolMatrix, primes: Vec<i32>) -> usize {
    let board_ints_vec = board
        .data()
        .iter()
        .map(|&cell| if cell { 1 } else { 0 })
        .collect::<Vec<i32>>();

    let ints_board = Matrix::from_vec(board_ints_vec, board.rows(), board.cols());

    let primes_board = Matrix::from_vec(primes, ints_board.rows(), ints_board.cols());

    let result = ints_board * primes_board;
    let result: i32 = result.data().iter().sum();
    result as usize
}

pub fn detect_repeating_state(board_hashes: &mut Vec<usize>) -> bool {
    // so - detect alternating states. if 0==2, 1==3, 2==4, 3==5, 4==6, 5==7
    if board_hashes.len() < 4 {
        return false;
    }
    let mut result = false;
    if (board_hashes[0] == board_hashes[2]) && (board_hashes[0] == board_hashes[2]) {
        result = true;
    }
    // remove the 0th item
    board_hashes.remove(0);
    result
}

pub fn add_simulated_activity(current_board: &mut BoolMatrix, board_size: usize) {
    for _ in 0..20 {
        generate_glider(current_board, board_size);
    }

    // Generate a Pulsar pattern
    for _ in 0..10 {
        generate_pulsar(current_board, board_size);
    }
}

// generate prime numbers
pub fn generate_primes(n: i32) -> Vec<i32> {
    // I want to generate the first n primes
    let mut primes = Vec::new();
    let mut count = 0;
    let mut num = 2; // Start checking for primes from 2
    while count < n {
        let mut is_prime = true;
        for i in 2..=((num as f64).sqrt() as i32) {
            if num % i == 0 {
                is_prime = false;
                break;
            }
        }
        if is_prime {
            primes.push(num);
            count += 1;
        }
        num += 1;
    }
    primes
}

// --- Tests from previous example (can be kept or adapted) ---
#[cfg(test)]
mod tests {
    use super::*;
    use rustframe::matrix::{BoolMatrix, BoolOps}; // Assuming BoolOps is available for .count()

    #[test]
    fn test_blinker_oscillator() {
        let initial_data = vec![false, true, false, false, true, false, false, true, false];
        let game1 = BoolMatrix::from_vec(initial_data.clone(), 3, 3);
        let expected_frame2_data = vec![false, false, false, true, true, true, false, false, false];
        let expected_game2 = BoolMatrix::from_vec(expected_frame2_data, 3, 3);
        let game2 = game_of_life_next_frame(&game1);
        assert_eq!(
            game2.data(),
            expected_game2.data(),
            "Frame 1 to Frame 2 failed for blinker"
        );
        let expected_game3 = BoolMatrix::from_vec(initial_data, 3, 3);
        let game3 = game_of_life_next_frame(&game2);
        assert_eq!(
            game3.data(),
            expected_game3.data(),
            "Frame 2 to Frame 3 failed for blinker"
        );
    }

    #[test]
    fn test_empty_board_remains_empty() {
        let board_3x3_all_false = BoolMatrix::from_vec(vec![false; 9], 3, 3);
        let next_frame = game_of_life_next_frame(&board_3x3_all_false);
        assert_eq!(
            next_frame.count(),
            0,
            "All-false board should result in all-false"
        );
    }

    #[test]
    fn test_zero_size_board() {
        let board_0x0 = BoolMatrix::from_vec(vec![], 0, 0);
        let next_frame = game_of_life_next_frame(&board_0x0);
        assert_eq!(next_frame.rows(), 0);
        assert_eq!(next_frame.cols(), 0);
        assert!(
            next_frame.data().is_empty(),
            "0x0 board should result in 0x0 board"
        );
    }

    #[test]
    fn test_still_life_block() {
        let block_data = vec![
            true, true, false, false, true, true, false, false, false, false, false, false, false,
            false, false, false,
        ];
        let game_block = BoolMatrix::from_vec(block_data.clone(), 4, 4);
        let next_frame_block = game_of_life_next_frame(&game_block);
        assert_eq!(
            next_frame_block.data(),
            game_block.data(),
            "Block still life should remain unchanged"
        );
    }
}
