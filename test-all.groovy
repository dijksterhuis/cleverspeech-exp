#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    triggers {
        upstream(upstreamProjects: './0-build/latest', threshold: hudson.model.Result.SUCCESS)
    }
    parameters {

        string name: 'MAX_SPAWNS',
            defaultValue: '2',
            description: 'Number of attacks to allow to spawn at once.'

        string name: 'BATCH_SIZE',
            defaultValue: '2',
            description: 'How many examples in a batch.'

        string name: 'N_STEPS',
            defaultValue: '100',
            description: 'How many iterations to run the attack for.'

        /* choice name: 'EXP_SCRIPT',
            choices: ['attacks', 'unbounded'],
            description: 'Which attack python script to run. default: attacks.py.'
        */

        choice name: 'DATA',
            choices: ['samples', 'silence'],
            description: 'Which dataset to use. default: ./samples'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '--decode_step 10 --max_examples 4 --spawn_delay 2 ',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'

    }
    stages {
        stage("Test one."){
            steps{
                echo "Starting baseline-ctc build job as an initial test..."
                build job: "baseline-ctc",
                    wait: true,
                    parameters: [
                        stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                        stringParam(name: 'EXP_SCRIPT', value: "attacks"),
                        stringParam(name: 'MAX_SPAWNS', value: "${params.MAX_SPAWNS}"),
                        stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                        stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                        stringParam(name: 'DATA', value: "${params.DATA}"),
                        stringParam(name: 'JOB_TYPE', value: "test"),
                    ]
            }
        }
        stage("Test others."){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'DIR'
                        values 'baseline-ctc',
                            'baseline-cwmaxdiff',
                            'baseline-biggiomaxmin',
                            'conf-adaptivekappa',
                            'conf-ctcedgecases',
                            'conf-invertedctc',
                            'conf-logprobsgreedydiff',
                            'conf-sumlogprobs',
                            'conf-cumulativelogprobs',
                            'conf-biggiomaxofmaxmin',
                            'conf-targetonly',
                            'misc-batch-vs-indy' /*,
                             'percep-synthesis',
                             'percep-synthesisregularised',
                             'percep-spectralloss'
                             */
                    }
                    axis {
                        name 'EXP_SCRIPT'
                        values 'attacks', 'unbounded'
                    }
                }
                /* exclude the baseline-ctc that already ran */
                excludes {
                    exclude {
                        axis {
                            name 'DIR'
                            values 'baseline-ctc'
                        }
                        axis {
                            name 'EXP_SCRIPT'
                            values 'attacks'
                        }
                    }
                }
                stages {
                    stage("Run experiment") {
                        steps {
                            echo "Starting ${DIR} build job for ${EXP_SCRIPT} script..."
                            build job: "${DIR}",
                                wait: true,
                                parameters: [
                                    stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                    stringParam(name: 'EXP_SCRIPT', value: "${EXP_SCRIPT}"),
                                    stringParam(name: 'MAX_SPAWNS', value: "${params.MAX_SPAWNS}"),
                                    stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                    stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                    stringParam(name: 'DATA', value: "${params.DATA}"),
                                    stringParam(name: 'JOB_TYPE', value: "test"),
                                ]
                        }
                    }
                }
            }
        }
    }
}