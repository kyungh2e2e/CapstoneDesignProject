#!/bin/bash

ks=(32 64 128 256 512 784)
runs=10
total_runs=10
epochs=30
model_type="perceiver-io-linstyle"

for k in "${ks[@]}"; do
    echo "=============================="
    echo "실험 시작: k = ${k}"
    echo "=============================="
    
    mkdir -p "./results/${k}"

    for (( test_no=0; test_no<runs; test_no++ )); do
        mkdir -p "./results/${k}/${test_no}"
        echo "[k=${k}] Run test_no=${test_no} 진행 중..."
        python main.py --model=${model_type} --k=${k} --test_no=${test_no} --epochs=${epochs} --batch_size=64 --lr=5e-4 --weight_decay=1e-4
    done

    echo "모든 run 완료! k=${k}에 대한 에포크별 Test Acc 평균 계산 시작..."


    report_file="./results/${k}/report_${k}.txt"
    echo "===== k = ${k} 에 대한 에포크별 평균 Test Accuracy =====" > "${report_file}"
    
    declare -a acc_sum
    declare -a acc_count
    for (( epoch=1; epoch<=epochs; epoch++ )); do
        acc_sum[$epoch]=0
        acc_count[$epoch]=0
    done

    total_time=0
    total_size=0
    time_count=0

    for (( test_no=0; test_no<total_runs; test_no++ )); do
        result_file="./results/${k}/${test_no}/perceiver_io_linstyle_results.txt"
        if [ -f "${result_file}" ]; then
            for (( epoch=1; epoch<=epochs; epoch++ )); do
                line=$(grep "Epoch ${epoch}:" "${result_file}")
                acc=$(echo "${line}" | awk -F"Test Acc=" '{print $2}' | xargs)
                if [[ ! -z "${acc}" ]]; then
                    acc_sum[$epoch]=$(echo "${acc_sum[$epoch]} + ${acc}" | bc -l)
                    acc_count[$epoch]=$((acc_count[$epoch] + 1))
                else
                    echo "경고: ${result_file}에서 epoch ${epoch}의 Test Acc 값을 읽지 못했습니다."
                fi
            done
            time_line=$(grep "총 학습 시간" "${result_file}")
            time_value=$(echo "${time_line}" | awk -F": " '{print $2}' | xargs)
            size_line=$(grep "모델 크기" "${result_file}")
            size_value=$(echo "${size_line}" | awk -F": " '{print $2}' | xargs)
            if [[ ! -z "${time_value}" && ! -z "${size_value}" ]]; then
                total_time=$(echo "${total_time} + ${time_value}" | bc -l)
                total_size=$(echo "${total_size} + ${size_value}" | bc -l)
                time_count=$((time_count + 1))
            else
                echo "경고: ${result_file}에서 학습 시간 또는 모델 크기 값을 읽지 못했습니다."
            fi
        else
            echo "경고: 결과 파일 ${result_file} 가 존재하지 않습니다."
        fi
    done

    for (( epoch=1; epoch<=epochs; epoch++ )); do
        if [ ${acc_count[$epoch]} -gt 0 ]; then
            avg=$(echo "scale=4; ${acc_sum[$epoch]} / ${acc_count[$epoch]}" | bc -l)
        else
            avg=0
        fi
        echo "Epoch ${epoch} AVG Acc: ${avg}" >> "${report_file}"
    done

    if [ ${time_count} -gt 0 ]; then
        avg_time=$(echo "scale=4; ${total_time} / ${time_count}" | bc -l)
        avg_size=$(echo "scale=4; ${total_size} / ${time_count}" | bc -l)
    else
        avg_time=0
        avg_size=0
    fi

    echo "Average Training Time (초): ${avg_time}" >> "${report_file}"
    echo "Average Model Size (MB): ${avg_size}" >> "${report_file}"
    echo "k=${k}에 대한 report 파일 생성 완료: ${report_file}"
    echo ""
done
