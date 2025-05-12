#!/bin/bash

ks=(32 64 128 256 512 784)
runs=10       
total_runs=10
epochs=30

model_type="perceiver-io-linstyle"

for k in "${ks[@]}"; do
    echo "=============================="
    echo "k = ${k} 에 대한 결과 집계 시작"
    echo "=============================="
    
    report_file="./results/${k}/report_${k}_avg.txt"
    mkdir -p "./results/${k}"
    echo "===== k = ${k} 에 대한 에포크별 평균 결과 =====" > "${report_file}"

    declare -a train_loss_sum
    declare -a test_loss_sum
    declare -a test_acc_sum
    declare -a count

    for (( epoch=1; epoch<=epochs; epoch++ )); do
        train_loss_sum[$epoch]=0
        test_loss_sum[$epoch]=0
        test_acc_sum[$epoch]=0
        count[$epoch]=0
    done

    for (( test_no=0; test_no<total_runs; test_no++ )); do
        result_file="./results/${k}/${test_no}/perceiver_io_linstyle_results.txt"
        if [ -f "${result_file}" ]; then
            for (( epoch=1; epoch<=epochs; epoch++ )); do
                line=$(grep "Epoch ${epoch}:" "${result_file}")
                train_loss=$(echo "${line}" | awk -F"Train Loss=" '{print $2}' | awk -F"," '{print $1}' | xargs)
                test_loss=$(echo "${line}" | awk -F"Test Loss=" '{print $2}' | awk -F"," '{print $1}' | xargs)
                test_acc=$(echo "${line}" | awk -F"Test Acc=" '{print $2}' | xargs)

                if [[ ! -z "${train_loss}" && ! -z "${test_loss}" && ! -z "${test_acc}" ]]; then
                    train_loss_sum[$epoch]=$(echo "${train_loss_sum[$epoch]} + ${train_loss}" | bc -l)
                    test_loss_sum[$epoch]=$(echo "${test_loss_sum[$epoch]} + ${test_loss}" | bc -l)
                    test_acc_sum[$epoch]=$(echo "${test_acc_sum[$epoch]} + ${test_acc}" | bc -l)
                    count[$epoch]=$((count[$epoch] + 1))
                else
                    echo "경고: ${result_file}의 epoch ${epoch} 정보를 읽을 수 없습니다."
                fi
            done
        else
            echo "경고: ${result_file} 파일 없음"
        fi
    done

    for (( epoch=1; epoch<=epochs; epoch++ )); do
        if [ ${count[$epoch]} -gt 0 ]; then
            avg_train_loss=$(echo "scale=4; ${train_loss_sum[$epoch]} / ${count[$epoch]}" | bc -l)
            avg_test_loss=$(echo "scale=4; ${test_loss_sum[$epoch]} / ${count[$epoch]}" | bc -l)
            avg_test_acc=$(echo "scale=4; ${test_acc_sum[$epoch]} / ${count[$epoch]}" | bc -l)
        else
            avg_train_loss=0
            avg_test_loss=0
            avg_test_acc=0
        fi
        echo "Epoch ${epoch}: AVG Train Loss=${avg_train_loss}, AVG Test Loss=${avg_test_loss}, AVG Test Acc=${avg_test_acc}" >> "${report_file}"
    done

    echo "===== k=${k} 에 대한 평균 report 생성 완료 ====="
    echo "파일 위치: ${report_file}"
    echo ""
done
