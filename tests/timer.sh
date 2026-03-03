#! /bin/bash

# sends inference requests in parallel

cnt=10
for _ in $(seq 1 $cnt); do
	curl -X POST localhost:9000/infer -d '{"question": "List 50 animals."}' &
done

wait
