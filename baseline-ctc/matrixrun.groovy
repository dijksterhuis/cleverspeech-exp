#!/usr/bin/env groovy

pipeline {
    /* Use jenkins build node to manage how many experiments to run at a time. */
    agent {
        label "build"
    }
    options {
        skipDefaultCheckout()
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    /*
    triggers {
        pollSCM('H H * * 1-5') }
        upstream(upstreamProjects: './build/latest', threshold: hudson.model.Result.SUCCESS) }
    }
    */
    environment {
        JOB_NAME = "baseline-ctc"
    }
    parameters {

        string name: 'N_STEPS',
            defaultValue: '10000',
            description: 'How many iterations to run the attack for.'

        string name: 'BATCH_SIZE',
            defaultValue: '10',
            description: 'How many examples in a batch.'

        choice name: 'JOB_TYPE',
            choices: ['run', 'test'],
            description: 'Whether this is an experiment run or if we are just testing that everything works. default: run.'

        choice name: 'EXP_SCRIPT',
            choices: ['attacks', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'

        choice name: 'DATA',
            choices: ['samples', 'silence'],
            description: 'Which dataset to use. default: ./samples'

        choice name: 'WRITER',
            choices: ['local', 's3'],
            description: 'How/where to write results data?. default: local.'

        choice name: 'LOSS_FILTER',
            choices: ['all', 'ctc', 'ctc2'],
            description: 'Filter experiments based on loss hyper parameter. Note that this only works when combined with other filters.'

        choice name: 'DECODER_FILTER',
            choices: ['all', 'batch', 'greedy', 'batch_no_lm', 'greedy_no_lm'],
            description: 'Filter experiments based on decoder hyper parameter. Note that this only works when combined with other filters.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'
    }

    stages {
        stage("Modify jenkins build information") {
            steps {
                script {
                    buildName "#${BUILD_ID}: type:${params.JOB_TYPE} script:${params.EXP_SCRIPT} data:${params.DATA} steps:${params.N_STEPS}"
                }
            }
        }
        stage("Run all combos in parallel."){
            failFast false /* If one run fails, keep going! */
            matrix {
                axes {
                    axis {
                        name 'LOSS'
                        values 'ctc', 'ctc2'
                    }
                    axis {
                        name 'DECODER'
                        values 'greedy', 'batch', 'batch_no_lm', 'greedy_no_lm'
                    }
                }
                stages {
                    stage("Run") {
                        steps {
                            build job: "../${JOB_NAME}", wait: true, propagate: true, parameters: [
                                stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                stringParam(name: 'EXP_SCRIPT', value: "${params.EXP_SCRIPT}"),
                                stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                stringParam(name: 'WRITER', value: "${params.WRITER}"),
                                stringParam(name: 'DATA', value: "${params.DATA}"),
                                stringParam(name: 'JOB_TYPE', value: "${params.JOB_TYPE}"),
                                stringParam(name: 'LOSS', value: "${LOSS}"),
                                stringParam(name: 'DECODER', value: "${DECODER}"),
                            ]
                        }
                    }
                }
            }
        }
    }
}
