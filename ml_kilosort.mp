{
	"processors":[
		{
			"name":"ml_kilosort",
			"version":"0.1",
			"description":"",
			"inputs":[{"name":"timeseries"}],
			"outputs":[{"name":"firings_out"}],
			"parameters":[
				{"name":"num_clusters","optional":false}
			],
			"exe_command":"$(basepath)/run_matlab.sh $(basepath)/run_kilosort \" run_mountainlab_setup; opts=struct('num_clusters',$num_clusters$); ml_kilosort('$timeseries$','$firings_out$',opts);\""
		}
	]
}
