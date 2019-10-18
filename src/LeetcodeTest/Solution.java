package LeetcodeTest;

public class Solution {

    // 用来标记该节点是否遍历过 因为只允许使用一次
    private boolean[][] marked;
    // 算出左上右下四个点坐标
    //          (x-1,y)
    // (x,y-1)   (x,y)  (x,y+1)
    //          (x+1,y)
    private int[][] directions = {{-1,0},{0,-1},{0,1},{1,0}};
    // 行数
    int m;
    // 列数
    int n;
    String word;
    private char[][] board;

    public boolean exist(char[][] board, String word) {
        m = board.length;
        if(m == 0){return false;}
        n = board[0].length;
        marked = new boolean[m][n];
        this.word = word;
        this.board = board;
        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                if(dfs(i,j,0)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean inArea(int x,int y){
        return x>=0 && x<m && y>=0 && y<n;
    }



    private boolean dfs(int i,int j,int start){
        if(start == word.length()-1){
            return board[i][j] == word.charAt(start);
        }
        if(board[i][j] == word.charAt(start)){
            marked[i][j] = true;
            for(int k = 0;k<4;k++){
                int newX = i + directions[k][0];
                int newY = j + directions[k][1];
                if(inArea(newX,newY) && !marked[newX][newY]){
                    if(dfs(newX,newY,start+1)){
                        return true;
                    }
                }

            }
            marked[i][j] = false;
        }
        return false;
    }

    public static void main(String[] args) {

        char[][] board =
                {
                        {'A', 'B', 'C', 'E'},
                        {'S', 'F', 'C', 'S'},
                        {'A', 'D', 'E', 'E'}
                };

        String word = "SEE";


//        char[][] board = {{'a', 'b'}};
//        String word = "ba";
        Solution solution = new Solution();
        boolean exist = solution.exist(board, word);
        System.out.println(exist);
    }
}

