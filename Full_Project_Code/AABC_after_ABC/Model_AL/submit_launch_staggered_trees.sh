#!/bin/bash

# number of AABC samples to run
samples=(
25000
50000
75000
100000
100001
100002
100003
100004
100005
)

previous_tree=""
delete_jobs=()

for SAMPLE in "${samples[@]}"
do

echo "Submitting jobs for $SAMPLE"

tree_dependency=""

if [ -n "$previous_tree" ]; then
  tree_dependency="done($previous_tree)"
fi

two_back_index=$((${#delete_jobs[@]} - 2))
if [ "$two_back_index" -ge 0 ]; then
  two_back_delete="${delete_jobs[$two_back_index]}"

  if [ -n "$tree_dependency" ]; then
    tree_dependency="${tree_dependency} && done($two_back_delete)"
  else
    tree_dependency="done($two_back_delete)"
  fi
fi

if [ -n "$tree_dependency" ]; then
  TREE=$(bsub -w "$tree_dependency" < <(
  sed "1a SAMPLE=$SAMPLE" submit_tree.sh
  ) | awk '{print $2}' | tr -d '<>')
else
  TREE=$(bsub < <(
  sed "1a SAMPLE=$SAMPLE" submit_tree.sh
  ) | awk '{print $2}' | tr -d '<>')
fi

STEP3=$(bsub -w "done($TREE)" < <(
sed "1a SAMPLE=$SAMPLE" submit_step3.sh
) | awk '{print $2}' | tr -d '<>')

DELETE=$(bsub -w "done($STEP3)" < <(
sed "1a SAMPLE=$SAMPLE" submit_delete.sh
) | awk '{print $2}' | tr -d '<>')

previous_tree="$TREE"
delete_jobs+=("$DELETE")

echo "  TREE job:  $TREE"
echo "  STEP3 job: $STEP3"
echo "  DELETE job:$DELETE"

done
