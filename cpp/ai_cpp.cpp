/*
 * ai_cpp.cpp -- Thin pybind11 wrapper around engine.h (opt variant).
 *
 * Build:  python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"

namespace py = pybind11;

// ── Helper: load patterns via Python ai module ──
static void load_patterns_from_python(opt::MinimaxBot& engine, py::object pattern_path) {
    py::module_ ai_mod = py::module_::import("minimax_bot");
    std::string path;
    if (pattern_path.is_none())
        path = ai_mod.attr("_DEFAULT_PATTERN_PATH").cast<std::string>();
    else
        path = pattern_path.cast<std::string>();

    py::tuple result = ai_mod.attr("_load_pattern_values")(path).cast<py::tuple>();
    py::list  pv_list = result[0].cast<py::list>();
    int       eval_length = result[1].cast<int>();

    std::vector<double> pv(pv_list.size());
    for (size_t i = 0; i < pv_list.size(); i++)
        pv[i] = pv_list[i].cast<double>();

    engine.load_patterns(pv, eval_length, path);
}

// ── Helper: extract GameState from Python game object ──
static GameState extract_game_state(py::object game) {
    py::module_ game_mod = py::module_::import("game");
    py::object  PyA = game_mod.attr("Player").attr("A");

    py::dict py_board = game.attr("board").cast<py::dict>();

    GameState gs;
    gs.cells.reserve(py_board.size());
    for (auto item : py_board) {
        py::tuple key = item.first.cast<py::tuple>();
        int q = key[0].cast<int>(), r = key[1].cast<int>();
        int8_t p = item.second.is(PyA) ? P_A : P_B;
        gs.cells.push_back({q, r, p});
    }

    py::object py_cur = game.attr("current_player");
    gs.cur_player  = py_cur.is(PyA) ? P_A : P_B;
    gs.moves_left  = game.attr("moves_left_in_turn").cast<int8_t>();
    gs.move_count  = game.attr("move_count").cast<int>();
    return gs;
}

// ── Wrapper class ──
struct PyMinimaxBot {
    opt::MinimaxBot engine;

    PyMinimaxBot() { load_patterns_from_python(engine, py::none()); }

    PyMinimaxBot(double tl, py::object pattern_path = py::none())
        : engine(tl)
    {
        load_patterns_from_python(engine, pattern_path);
    }

    py::list get_move(py::object game) {
        auto gs = extract_game_state(game);
        if (gs.cells.empty()) {
            py::list res;
            res.append(py::make_tuple(0, 0));
            return res;
        }

        auto mr = engine.get_move(gs);

        py::list res;
        res.append(py::make_tuple(mr.q1, mr.r1));
        if (mr.num_moves > 1)
            res.append(py::make_tuple(mr.q2, mr.r2));
        return res;
    }

    // Init board state without searching (for pre-filter checks)
    void init_board(py::object game) {
        auto gs = extract_game_state(game);
        // Use get_move internals: clear, populate, init windows/candidates
        // but with a near-zero time limit so it returns immediately after depth 1
        double saved_tl = engine.time_limit;
        engine.time_limit = 0.0001;
        engine.get_move(gs);
        engine.time_limit = saved_tl;
    }

    bool has_instant_win() const { return engine.has_instant_win(); }
    bool has_near_threats() const { return engine.has_near_threats(); }

    py::list extract_pv() {
        auto pv = engine.extract_pv();
        py::list result;
        for (const auto& step : pv) {
            py::dict d;
            d["player"] = (step.player == P_A) ? "A" : "B";
            py::list moves;
            for (const auto& [q, r] : step.moves)
                moves.append(py::make_tuple(q, r));
            d["moves"] = moves;
            result.append(d);
        }
        return result;
    }

    py::tuple getstate() const {
        auto es = engine.get_state();
        py::bytes pv_bytes(reinterpret_cast<const char*>(es.pv.data()),
                           es.pv.size() * sizeof(double));
        return py::make_tuple(es.time_limit, pv_bytes,
                              static_cast<int>(es.pv.size()),
                              es.eval_length, es.pattern_path_str);
    }

    void setstate(py::tuple t) {
        EngineState es;
        es.time_limit       = t[0].cast<double>();
        auto pv_str         = t[1].cast<std::string>();
        int  pv_size        = t[2].cast<int>();
        es.eval_length      = t[3].cast<int>();
        es.pattern_path_str = t[4].cast<std::string>();
        es.pv.resize(pv_size);
        std::memcpy(es.pv.data(), pv_str.data(), pv_size * sizeof(double));
        engine.set_state(es);
    }
};

// ═══════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(ai_cpp, m) {
    m.doc() = "C++ port of ai.py MinimaxBot (flat-array board)";

    py::class_<PyMinimaxBot>(m, "MinimaxBot")
        .def(py::init<double, py::object>(),
             py::arg("time_limit") = 0.05,
             py::arg("pattern_path") = py::none())
        .def("get_move", &PyMinimaxBot::get_move, py::arg("game"))
        .def("init_board", &PyMinimaxBot::init_board, py::arg("game"))
        .def("has_instant_win", &PyMinimaxBot::has_instant_win)
        .def("has_near_threats", &PyMinimaxBot::has_near_threats)
        .def("extract_pv", &PyMinimaxBot::extract_pv)
        .def_property("pair_moves",
            [](PyMinimaxBot& b) { return b.engine.pair_moves; },
            [](PyMinimaxBot& b, bool v) { b.engine.pair_moves = v; })
        .def_property("no_cand_cap",
            [](PyMinimaxBot& b) { return b.engine.no_cand_cap; },
            [](PyMinimaxBot& b, bool v) { b.engine.no_cand_cap = v; })
        .def_property("time_limit",
            [](PyMinimaxBot& b) { return b.engine.time_limit; },
            [](PyMinimaxBot& b, double v) { b.engine.time_limit = v; })
        .def_property("last_depth",
            [](PyMinimaxBot& b) { return b.engine.last_depth; },
            [](PyMinimaxBot& b, int v) { b.engine.last_depth = v; })
        .def_property("_nodes",
            [](PyMinimaxBot& b) { return b.engine._nodes; },
            [](PyMinimaxBot& b, int v) { b.engine._nodes = v; })
        .def_property("last_score",
            [](PyMinimaxBot& b) { return b.engine.last_score; },
            [](PyMinimaxBot& b, double v) { b.engine.last_score = v; })
        .def_property("last_ebf",
            [](PyMinimaxBot& b) { return b.engine.last_ebf; },
            [](PyMinimaxBot& b, double v) { b.engine.last_ebf = v; })
        .def_property("max_depth",
            [](PyMinimaxBot& b) { return b.engine.max_depth; },
            [](PyMinimaxBot& b, int v) { b.engine.max_depth = v; })
        .def("__str__", [](const PyMinimaxBot&) { return std::string("ai_cpp"); })
        .def(py::pickle(
            [](const PyMinimaxBot& bot) { return bot.getstate(); },
            [](py::tuple t) {
                PyMinimaxBot bot;
                bot.setstate(t);
                return bot;
            }
        ));
}
