package NEW_JOURNEY_IN_WORK.array;

public class LC_832 {
//    public static int[][] flipAndInvertImage(int[][] A) {
//        int m = A.length;
//        int n = A[0].length;
//        for(int i = 0; i < m;i++){
//            int p = 0,q = n-1;
//            while(p<=q){
//                int tmp = A[i][q];
//                A[i][q] = A[i][p]==0?1:0;
//                A[i][p] = tmp == 0?1:0;
//                p++;
//                q--;
//            }
//        }
//        return A;
//    }

    /**
     * a   b   a⊕b
     * 1   0    1
     * 1   1    0
     * 0   0    0
     * 0   1    1
     *
     * 异或：两个输入相同时为0，不同则为1
     * 和1异或表示取反，和0异或 值与本身相同
     * @param A
     * @return
     */
    public static int[][] flipAndInvertImage(int[][] A) {
        int C = A[0].length;
        for (int[] row : A)
            // for (int i = 0; i < (C + 1) / 2; ++i)
            // 这句话对于偶数数组遍历一半元素，对于奇数数组遍历一半 + 1元素
            for (int i = 0; i < (C + 1) / 2; ++i) {
                int tmp = row[i] ^ 1;
                row[i] = row[C - 1 - i] ^ 1;
                row[C - 1 - i] = tmp;
            }

        return A;
    }


    public static void main(String[] args) {
        flipAndInvertImage(new int[][]{{1, 1, 0}, {1, 0, 1}, {0, 0, 0}});
    }
}
