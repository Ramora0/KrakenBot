/*
 * types.h -- Platform-neutral interface types shared by all engine
 *            variants and all wrappers (pybind11, Emscripten, etc.).
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Player identifiers (match game.Player enum values)
static constexpr int8_t P_NONE = 0;
static constexpr int8_t P_A    = 1;
static constexpr int8_t P_B    = 2;

// Board state passed into the engine (no Python / JS types).
struct GameState {
    struct Cell { int q, r; int8_t player; };   // player = P_A or P_B
    std::vector<Cell> cells;
    int8_t cur_player;   // P_A or P_B
    int8_t moves_left;   // 1 or 2
    int    move_count;   // total stones on board
};

// Engine result: the best pair of moves found.
struct MoveResult {
    int q1, r1, q2, r2;
    int num_moves;       // 1 (edge-case / first turn) or 2 (normal)
};

// Serialisable snapshot for pickle / restore.
struct EngineState {
    double              time_limit;
    std::vector<double> pv;
    int                 eval_length;
    std::string         pattern_path_str;
};
