import argparse




def parse_cli_args():

	# Create the parser
	my_parser = argparse.ArgumentParser(description='event detector')

	my_parser.add_argument('-k',
                       	   '--kink_factor',
                           metavar='kink factor',
                           default=0.4,
                           type=float,
                           help='It is coefficient (usually between 0 and 1) to decides the event initiating point (local maximum change is not a physiologically starting timing). Default = 0.4.')

	my_parser.add_argument('-i',
	                       '--diff_window',
	                       metavar='interval for differentiation',
	                       default=0.3,
	                       type=float,
	                       help='The time interval (s) for differentiating the trace. It would affect the decision of diff_thrsld. Default = 0.3.')

	my_parser.add_argument('-r',
	                       '--raw_thrsld',
	                       metavar='threshold on raw value',
	                       default=0.55,
	                       type=float,
	                       help='An event detection criteria. The normalized amplitude of an event should higher than this. Default = 0.55.')

	my_parser.add_argument('-d',
	                       '--diff_thrsld',
	                       metavar='threshold on differentiated value',
	                       default=0.2,
	                       type=float,
	                       help='An event detection criteria. The differentiated normalized amplitude of an event should higher than this. The choice of number can be affected by diff_window. Default = 0.2.')

	my_parser.add_argument('-sd',
	                       '--shortest_evt_dur',
	                       metavar='shortest event duration',
	                       default=0.5,
	                       type=float,
	                       help='An event detection criteria. The duration of an event should longer than this (s). Default = 0.5.')

	my_parser.add_argument('-ld',
	                       '--longest_evt_dur',
	                       metavar='longest event duration',
	                       default=2,
	                       type=float,
	                       help='Not an event detection criteria. The duration of an event (s). Default = 2.')

	my_parser.add_argument('-input',
	                       metavar='input file',
	                       default='./data/trial001_0.csv',
	                       type=str,
	                       help='input directory + filename.csv')

	my_parser.add_argument('-output',
	                       metavar='output directory',
	                       default='./output_events/',
	                       type=str,
	                       help='output directory')		

	my_parser.add_argument('-plot_overlay',
	                       metavar='plot overlaid events',
	                       default=True,
	                       type=bool,
	                       help='an optional plot of overlaid detected events. Default = True.')		


	# Execute the parse_args() method
	args = my_parser.parse_args()

	return args