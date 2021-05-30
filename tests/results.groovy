#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
        skipDefaultCheckout()
    }
    triggers {
        upstream(upstreamProjects: './evasion', threshold: hudson.model.Result.SUCCESS)
    }
    parameters {

        string name: 'BATCH_SIZE',
            defaultValue: '2',
            description: 'How many examples in a batch.'

        string name: 'N_STEPS',
            defaultValue: '500',
            description: 'How many iterations to run the attack for.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '--decode_step 50 --max_examples 4',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10.'

    }
    stages {
        stage("SCM") {
            steps {
                lock("dummy") {
                    checkout scm
                }
            }
        }
        stage("Tests"){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'SCRIPT'
                        values 'attacks', 'unbounded'
                    }
                }
                stages {
                    stage("t") {
                        steps {
                            echo "Starting a minimal ctc attack build job for ${SCRIPT}"
                            build job: "../baseline-ctc",
                                wait: true, propagate: true,
                                parameters: [
                                    stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                    stringParam(name: 'EXP_SCRIPT', value: "${SCRIPT}"),
                                    stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                    stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                    stringParam(name: 'WRITER', value: "local_latest"),
                                    stringParam(name: 'DATA', value: "samples"),
                                    stringParam(name: 'JOB_TYPE', value: "test"),
                                ]
                        }
                    }
                }
            }
        }
    }
}