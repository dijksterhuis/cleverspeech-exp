#!/usr/bin/env groovy

pipeline {
    agent { label 'build' }
    stages {
        stage("Run all the experiments!"){
            failFast false
            matrix {
                axes {
                    axis {
                        name 'DIR'
                        values 'CTCBaselines', 'Confidence/CWMaxDiffBaselines', 'Confidence/AdaptiveKappa', 'Confidence/AlignmentEdgeCases', 'Confidence/InvertedCTC', 'Confidence/Vibertish', 'Perceptual/Synthesis', 'Perceptual/RegularisedSynthesis', 'Perceptual/SpectralLossRegularisation'
                    }
                }
                stages {
                    stage("Run experiment") {
                        steps {
                            echo "Starting ${DIR} build job..."
                            build job: "${DIR}", wait: false
                        }
                    }
                }
            }
        }
    }
}