#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    options {
        timestamps()
        disableResume()
        disableConcurrentBuilds()
    }
    triggers {
        upstream(upstreamProjects: './unbounded', threshold: hudson.model.Result.SUCCESS)
    }
    parameters {

        string name: 'BATCH_SIZE',
            defaultValue: '2',
            description: 'How many examples in a batch.'

        string name: 'N_STEPS',
            defaultValue: '10',
            description: 'How many iterations to run the attack for.'

        text   name: 'ADDITIONAL_ARGS',
            defaultValue: '--decode_step 2 --max_examples 4',
            description: 'Additional arguments to pass to the attack script e.g. --decode_step 10. default: none.'

    }
    stages {
        stage("Test one."){
            steps{
                echo "Starting baseline-ctc build job as an initial test..."
                build job: "../baseline-ctc",
                    wait: true,
                    parameters: [
                        stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                        stringParam(name: 'EXP_SCRIPT', value: "attacks"),
                        stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                        stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                        stringParam(name: 'WRITER', value: "local"),
                        stringParam(name: 'DATA', value: "samples"),
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
                            'conf-weightedmaxmin',
                            'conf-maxadvctc-mintruectc',
                            'misc-batch-vs-indy' /*,
                             'percep-synthesis',
                             'percep-synthesisregularised',
                             'percep-spectralloss'
                             */
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
                    stage("t") {
                        steps {
                            echo "Starting ${DIR} build job"
                            build job: "../${DIR}",
                                wait: true, propagate: true,
                                parameters: [
                                    stringParam(name: 'ADDITIONAL_ARGS', value: "${params.ADDITIONAL_ARGS}"),
                                    stringParam(name: 'EXP_SCRIPT', value: "attacks"),
                                    stringParam(name: 'BATCH_SIZE', value: "${params.BATCH_SIZE}"),
                                    stringParam(name: 'N_STEPS', value: "${params.N_STEPS}"),
                                    stringParam(name: 'WRITER', value: "local"),
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