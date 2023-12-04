#tamanho da sequencia
tam_seq = 16

#pasta para redirecionar arqs de saída
path_seqs = 'Seqs' + str(tam_seq) + '/'

#arquivo de entrada com o caminho das imagens
arq_entrada = 'flare_Mclass_48h_Test.txt'
arq = open(arq_entrada)
linhas = arq.readlines()

#nome para o arquivo de saída
arq_saida = 'Seq' + str(tam_seq) + '_' + arq_entrada

#zero init
ra_ant = '0'
seq_atual = []
seq_virtual = []

for linha in linhas:
    linha_res = linha.split()
    #print(linha)
    #print(linha_res[0])
    #print(linha_res[1])

    current_img = linha.split('.')
    #print(current_img)
    ra = current_img[2]
    #print(type(ra))

    if ra != ra_ant:
        print("limpou")
        seq_atual.clear()
        ra_ant = ra
        #seq_init = current_img[3]

    seq_atual.append(linha_res[0])

    if len(seq_atual) > tam_seq:
        del seq_atual[0]

    first_img = seq_atual[0].split('.')
    last_img = seq_atual[-1].split('.')
    print(first_img)
    print(last_img)

    filename = last_img[0] + '.' + last_img[1] + '.' + last_img[2] + '.' + first_img[3] + '.to.' + last_img[3]
    print(filename)

    if len(seq_atual) == tam_seq:
        with open(path_seqs + filename, 'w') as fout:
            fout.write('\n'.join(f'{img}' for img in seq_atual))
    else:
        seq_virtual.clear()
        mult_aloc = len(seq_atual) / tam_seq
        for i in range(tam_seq):
            j = int(i * mult_aloc)
            seq_virtual.append(seq_atual[j])

        with open(path_seqs + filename, 'w') as fout:
            fout.write('\n'.join(f'{img}' for img in seq_virtual))

    with open(arq_saida, 'a') as fout:
        fout.write(filename + ' ' + linha_res[1] + '\n')

    #print(seq_atual)
