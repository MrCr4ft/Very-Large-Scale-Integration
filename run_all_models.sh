#!/bin/bash

# Run all models
# Usage: ./run_all_models.sh <instances_dir> <solutions_dir> <time_limit>

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: ./run_all_models.sh <instances_dir> <solutions_dir> <time_limit>"
    exit 1
fi

# Get arguments
INSTANCES_DIR=$1
SOLUTIONS_DIR=$2
TIME_LIMIT=$3

# Create solutions directory
mkdir -p $SOLUTIONS_DIR

#CP with Chuffed

#Best model without rotation with symmetry breaking 1
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb_1.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb_1/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb_1/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 1
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb_1.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb_1/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb_1/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True


#Best model without rotation with symmetry breaking 2
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb_2.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb_2/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb_2/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 2
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb_2.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb_2/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb_2/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True


#Best model without rotation without symmetry breaking
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_no_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_no_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation without symmetry breaking
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Board channeling without rotation
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=False
#Board channeling with rotation
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling_rotation.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling_rotation/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling_rotation/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=False

#CP with Gecode

#Best model without rotation with symmetry breaking 1
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb_1.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb_1/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb_1/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 1
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb_1.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb_1/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb_1/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True


#Best model without rotation with symmetry breaking 2
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb_2.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb_2/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb_2/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 2
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb_2.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb_2/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb_2/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True


#Best model without rotation without symmetry breaking
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_no_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_no_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation without symmetry breaking
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Board channeling without rotation
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=False
#Board channeling with rotation
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling_rotation.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling_rotation/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling_rotation/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=False

# MILP models with GUROBI
#SGBMRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=GUROBI_CMD
#SGBMNoRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=GUROBI_CMD
#S1BMRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s1_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s1_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S1BMRotation --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#S1BMNoRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s1_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s1_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S1BM --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD --solver-for-milp=GUROBI_CMD
#S2BMRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s2_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s2_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S2BMRotation --presolve-for-milp=True --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#S2BMNoRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s2_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s2_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S2BM --presolve-for-milp=True --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#PCMILPNoRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPNoRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#PCMILPRotationGUROBI
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD

# MILP models with CPLEX
#SGBMRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=CPLEX_CMD
#SGBMNoRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=CPLEX_CMD
#S1BMRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s1_with_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s1_with_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S1BMRotation --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#S1BMNoRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s1_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s1_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S1BM --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD --solver-for-milp=CPLEX_CMD
#S2BMRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s2_with_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s2_with_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S2BMRotation --presolve-for-milp=True --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#S2BMNoRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/s2_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/s2_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=S2BM --presolve-for-milp=True --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#PCMILPNoRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPNoRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#PCMILPRotationCPLEX
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD

#SMT

#SMTNoRotationWithZ3
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/smt_no_rotation_z3" --stats-output-csv-file="$SOLUTIONS_DIR/smt_no_rotation_z3/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SMT --activate-symmetry-breaking=True --draw-solutions=False --solver-for-smt=z3 --enable-cumulative-constraints-smt=True --allow-rotation-smt=False
#SMTNoRotationWithCVC5
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/smt_no_rotation_cvc5" --stats-output-csv-file="$SOLUTIONS_DIR/smt_no_rotation_cvc5/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SMT --activate-symmetry-breaking=True --draw-solutions=False --solver-for-smt=cvc5 --enable-cumulative-constraints-smt=True --allow-rotation-smt=False
#SMTRotationWithZ3
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/smt_with_rotation_z3" --stats-output-csv-file="$SOLUTIONS_DIR/smt_with_rotation_z3/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SMT --activate-symmetry-breaking=True --draw-solutions=False --solver-for-smt=z3 --enable-cumulative-constraints-smt=True --allow-rotation-smt=True
#SMTRotationWithCVC5
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/smt_with_rotation_cvc5" --stats-output-csv-file="$SOLUTIONS_DIR/smt_with_rotation_cvc5/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SMT --activate-symmetry-breaking=True --draw-solutions=False --solver-for-smt=cvc5 --enable-cumulative-constraints-smt=True --allow-rotation-smt=True

#SAT

#SATNoRotation
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_no_rotation" --stats-output-csv-file="$SOLUTIONS_DIR/sat_no_rotation/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATNoRotation --activate-symmetry-breaking=True --draw-solutions=False
#SATRotation
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_with_rotation" --stats-output-csv-file="$SOLUTIONS_DIR/sat_with_rotation/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATRotation --activate-symmetry-breaking=True --draw-solutions=False
