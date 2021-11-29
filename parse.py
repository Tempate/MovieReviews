from optparse import OptionParser
import re


def main():
    options = read_commands()

    label_to_key = {}

    with open(options.output, 'w') as output:
        for entry in options.data:
            expr = re.match('"(.*)",(\S+)', entry)
            
            if expr:
                text  = expr.group(1)
                label = expr.group(2)

                try:
                    key = label_to_key[label]
                except KeyError:
                    key = len(label_to_key)
                    label_to_key[label] = key

                output.write('"%s",%d\n' % (text, key))



def read_commands():
    parser = OptionParser("%prog -d <dataset> -o <output>")
    parser.add_option("-d", dest="dataset", help="Dataset to parse")
    parser.add_option("-o", dest="output", help="Output file")

    options, args = parser.parse_args()

    if not options.dataset or not options.output:
        parser.print_help()
        exit(0)

    with open(options.dataset) as dataset:
        options.data = dataset.read().splitlines()

    return options


if __name__ == "__main__":
    main()
