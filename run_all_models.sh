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

#Best model without rotation with symmetry breaking 1 (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 1 (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Best model without rotation without symmetry breaking (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_no_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_no_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation without symmetry breaking (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Board channeling without rotation (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=False
#Board channeling with rotation (TODO)
python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling_rotation.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling_rotation/chuffed/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling_rotation/chuffed/stats.csv" --solver="chuffed" --timeout-ms=$TIME_LIMIT --draw-solutions=False

#CP with Gecode

#Best model without rotation with symmetry breaking 1 (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation with symmetry breaking 1 (DONE)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Best model without rotation without symmetry breaking (NOT REALLY NEEDED)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_no_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_no_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True
#Best model with rotation without symmetry breaking (NOT REALLY NEEDED)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/best_model_rotation_no_sb.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_best_model_rotation_no_sb/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=True

#Board channeling without rotation (NOT REALLY NEEDED)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=False
#Board channeling with rotation (NOT REALLY NEEDED)
#python ./solve_instances_minizinc.py --instances-input-dir="$INSTANCES_DIR/dzn" --model-filepath="./cp/board_channeling_rotation.mzn" --solutions-output-dir="$SOLUTIONS_DIR/cp_board_channeling_rotation/gecode/" --stats-output-csv-file="$SOLUTIONS_DIR/cp_board_channeling_rotation/gecode/stats.csv" --solver="gecode" --timeout-ms=$TIME_LIMIT --draw-solutions=False



#SAT

#SATNoRotation With Symmetry Breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_no_rotation_sb" --stats-output-csv-file="$SOLUTIONS_DIR/sat_no_rotation_sb/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATNoRotation --activate-symmetry-breaking=True --draw-solutions=False
#SATRotation With Symmetry Breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_with_rotation_sb" --stats-output-csv-file="$SOLUTIONS_DIR/sat_with_rotation_sb/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATRotation --activate-symmetry-breaking=True --draw-solutions=False

#SATNoRotation Without Symmetry Breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_no_rotation" --stats-output-csv-file="$SOLUTIONS_DIR/sat_no_rotation/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATNoRotation --activate-symmetry-breaking=False --draw-solutions=False
#SATRotation Without Symmetry Breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/sat_with_rotation" --stats-output-csv-file="$SOLUTIONS_DIR/sat_with_rotation/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SATRotation --activate-symmetry-breaking=False --draw-solutions=False



# MILP models with GUROBI

#SGBMRotation No symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=GUROBI_CMD
#SGBMNoRotation No symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=GUROBI_CMD
#SGBMRotation With symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation_sb/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation_sb/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#SGBMNoRotation With symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation_sb/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation_sb/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD


#PCMILPNoRotation
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPNoRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD
#PCMILPRotation
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/gurobi" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_with_rotation/gurobi/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=GUROBI_CMD

# MILP models with CPLEX

#SGBMRotation No symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=CPLEX_CMD
#SGBMNoRotation No symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=False --draw-solutions=False --solver-for-milp=CPLEX_CMD
#SGBMRotation With symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_with_rotation_sb/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_with_rotation_sb/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBMRotation --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#SGBMNoRotation With symmetry breaking
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/sg_no_rotation_sb/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/sg_no_rotation_sb/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=SGBM --presolve-for-milp=False --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD


#PCMILPNoRotation
python ./solve_instances_sat_milp_smt.py --instances-input-dir="$INSTANCES_DIR/json" --solutions-output-dir="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/cplex" --stats-output-csv-file="$SOLUTIONS_DIR/milp/positioning_and_covering_no_rotation/cplex/stats.csv" --timeout-ms=$TIME_LIMIT --model-name=PCMILPNoRotation --activate-symmetry-breaking=True --draw-solutions=False --solver-for-milp=CPLEX_CMD
#PCMILPRotation
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