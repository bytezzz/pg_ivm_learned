#!/bin/bash

function select_process_and_attach() {
    local processes=$(ps -ef | grep "postgres: " | grep -v grep | awk '{print $2 " " $8 " " $9 " " $10 " " $11 " " $12 " " $13 " " $14 " " $15}')
    local IFS=$'\n'
    local options=($processes)
    local PS3='Please select a backend process to attach with gdb: '

    select opt in "${options[@]}" "Quit"; do
        case $opt in
            "Quit")
                echo "Exiting script."
                exit 0
                ;;
            *)
                local pid=$(echo $opt | awk '{print $1}')
                if [[ -n $pid ]]; then
                    echo "Attaching gdb to process $pid"
                    gdb -p $pid
                    break
                else
                    echo "Invalid selection."
                fi
                ;;
        esac
    done
}

function main() {
    select_process_and_attach
}

main
