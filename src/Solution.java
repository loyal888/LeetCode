import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Stack;

public class Solution {


    /**
     * 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
     * 每一列都按照从上到下递增的顺序排序。
     * 请完成一个函数，输入这样的一个二维数组和一个整数
     * ，判断数组中是否含有该整数。
     *
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int[][] array) {
        // array 为空
        if (array.length == 0) {
            return false;
        }
        // 获取行数
        int row = array.length;
        // 获取列数
        int col = array[0].length;
        // 一直循环
        int i = 0, j = col - 1;
        while (i < row && j >= 0) {
            if (array[i][j] > target) {
                j--;
            } else if (array[i][j] < target) {
                i++;
            } else {
                return true;
            }
        }
        return false;
    }


    /**
     * 请实现一个函数，将一个字符串中的每个空格替换成“%20”。
     * 例如，当字符串为We Are Happy.
     * 则经过替换之后的字符串为We%20Are%20Happy。
     *
     * @param str
     * @return
     */
    public String replaceSpace(StringBuffer str) {
        return str.toString().replace(" ", "%20");
    }

    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }

        /**
         * 输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
         *
         * @param listNode
         * @return
         */
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            ArrayList<Integer> arrayList = new ArrayList<Integer>();
            // 传入的是空节点
            if (listNode == null) {
                return arrayList;
            }
            ListNode L = null;
            ListNode M = null;
            ListNode R = listNode;
            // 先反转链表
            while (R.next != null) {
                L = M;
                M = R;
                R = R.next;
                M.next = L;
            }
            R.next = M;
            // 遍历取值
            while (R.next != null) {
                arrayList.add(R.val);
                R = R.next;
            }
            arrayList.add(R.val);
            return arrayList;
        }

    }

    // 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);

    }


    public int pop() {
        int data = 0;
        // 先把stack1中的值搬到stack2中
        while (!stack1.empty()) {
            stack2.push(stack1.pop());
        }
        // 取出栈顶数据
        data = stack2.pop();
        // stack2 中的数据放回
        while (!stack2.empty()) {
            stack1.add(stack2.pop());
        }
        // 返回data
        return data;
    }

    /**
     * 大家都知道斐波那契数列，现在要求输入一个整数n，
     * 请你输出斐波那契数列的第n项（从0开始，第0项为0）。
     * n<=39
     *
     * @param n 斐波那契数列的第n项
     * @return
     */
    public int Fibonacci(int n) {
        if (n < 0) {
            System.out.print("n can not 小于 0");
            return 0;
        }
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        return Fibonacci(n - 1) + Fibonacci(n - 2);

    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法
     * （先后次序不同算不同的结果）。
     * <p>
     * 解: 第一次跳一级时，f(1)= 1;余下f(n-1)
     * 第一次跳二级时，f(2)= 1;余下f(n-2)
     * f(n) = f(n-1)+f(n-2)
     *
     * @param target
     * @return
     */
    public int JumpFloor(int target) {
        if (target < 0) {
            System.out.print("target can not 小于 0");
            return 0;
        }
        if (target == 0) {
            return 0;
        }
        if (target == 1) {
            return 1;
        }
        if (target == 2) {
            return 2;
        }
        return JumpFloor(target - 1) + JumpFloor(target - 2);

    }

    /**
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级……
     * 它也可以跳上n级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法。
     * <p>
     * 解： 关于本题，前提是n个台阶会有一次n阶的跳法。分析如下:
     * f(1) = 1
     * f(2) = f(2-1) + f(2-2)         //f(2-2) 表示2阶一次跳2阶的次数。
     * f(3) = f(3-1) + f(3-2) + f(3-3)..
     * f(n) = f(n-1) + f(n-2) + f(n-3) + ... + f(n-(n-1)) + f(n-n)
     * 说明：
     * 1）这里的f(n) 代表的是n个台阶有一次1,2,...n阶的 跳法数。
     * 2）n = 1时，只有1种跳法，f(1) = 1
     * 3) n = 2时，会有两个跳得方式，一次1阶或者2阶，这回归到了问题（1） ，f(2) = f(2-1) + f(2-2)
     * 4) n = 3时，会有三种跳得方式，1阶、2阶、3阶，
     * 那么就是第一次跳出1阶后面剩下：f(3-1);第一次跳出2阶，剩下f(3-2)；第一次3阶，那么剩下f(3-3)
     * 因此结论是f(3) = f(3-1)+f(3-2)+f(3-3)
     * 5) n = n时，会有n中跳的方式，1阶、2阶...n阶，得出结论：
     * f(n) = f(n-1)+f(n-2)+...+f(n-(n-1)) + f(n-n) => f(0) + f(1) + f(2) + f(3) + ... + f(n-1)
     * 6) 由以上已经是一种结论，但是为了简单，我们可以继续简化：
     * f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
     * f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)
     * 可以得出：
     * f(n) = 2*f(n-1)
     * 7) 得出最终结论,在n阶台阶，一次有1、2、...n阶的跳的方式时，总得跳法为：
     * <p>
     * | 1       ,(n=0 )
     * <p>
     * f(n) =     | 1       ,(n=1 )
     * <p>
     * | 2*f(n-1),(n>=2)
     *
     * @param target
     * @return
     */
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return -1;
        } else if (target == 1) {
            return 1;
        } else {
            return 2 * JumpFloorII(target - 1);
        }
    }


    /**
     * 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
     * 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，
     * 总共有多少种方法？
     * 解：
     * 依旧是斐波那契数列
     * 2*n的大矩形，和n个2*1的小矩形
     * 其中target*2为大矩阵的大小
     * 有以下几种情形：
     * 1⃣️target <= 0 大矩形为<= 2*0,直接return 1；
     * 2⃣️target = 1大矩形为2*1，只有一种摆放方法，return1；
     * 3⃣️target = 2 大矩形为2*2，有两种摆放方法，return2；
     * 4⃣️target = n 分为两步考虑：
     * 第一次摆放一块 2*1 的小矩阵，则摆放方法总共为f(target - 1)
     * √
     * √
     * 第一次摆放一块1*2的小矩阵，则摆放方法总共为f(target-2)
     * 因为，摆放了一块1*2的小矩阵（用√√表示），对应下方的1*2（用××表示）摆放方法就确定了，所以为f(targte-2)
     * √	√
     * ×	×
     *
     * @param target
     * @return
     */
    public int RectCover(int target) {
        if (target <= 1) {
            return 1;
        }
        if (target * 2 == 2) {
            return 1;
        } else if (target * 2 == 4) {
            return 2;
        } else {
            return RectCover((target - 1)) + RectCover(target - 2);
        }
    }

    /**
     * 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
     * 解：
     * 如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，
     * 原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
     * 举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，
     * 它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.
     * 我们发现减1的结果是把最右边的一个1开始的所有位都取反了。
     * 这个时候如果我们再把原来的整数和减去1之后的结果做与运算，
     * 从原来整数最右边一个1那一位开始所有位都会变成0。
     * 如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，
     * 会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。
     *
     * @param n
     * @return
     */
    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;
    }

    /**
     * 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
     * 保证base和exponent不同时为0
     * <p>
     * 1.全面考察指数的正负、底数是否为零等情况。
     * 2.写出指数的二进制表达，例如13表达为二进制1101。
     * 3.举例:10^1101 = 10^0001*10^0100*10^1000。
     * 4.通过&1和>>1来逐位读取1101，为1时将该位代表的乘数累乘到最终结果。
     */
    public double Power(double base, int n) {
        double res = 1, curr = base;
        int exponent;
        if (n > 0) {
            exponent = n;
        } else if (n < 0) {
            if (base == 0) {
                throw new RuntimeException("分母不能为0");
            }
            exponent = -n;
        } else {// n==0
            return 1;// 0的0次方
        }
        while (exponent != 0) {
            if ((exponent & 1) == 1) {
                res *= curr;
            }
            curr *= curr;// 翻倍
            exponent >>= 1;// 右移一位
        }
        return n >= 0 ? res : (1 / res);
    }


    /**
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
     * 使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
     * 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
     *
     * @param array
     */
    public void reOrderArray(int[] array) {
        //相对位置不变，稳定性
        //插入排序的思想
        // 数组长度
        int length = array.length;
        // 用来记录奇数的个数
        int k = 0;
        for (int i = 0; i < length; i++) {
            int j = i;
            if (array[i] % 2 == 1) {
                while (j > k) {
                    int temp = array[j];
                    array[j] = array[j - 1];
                    array[j - 1] = temp;
                    j--;
                }
                k++;
            }
        }
    }

    /**
     * 输入一个链表，输出该链表中倒数第k个结点。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null) {
            return head;
        }
        int length = 1;
        ListNode p = head;
        while (p.next != null) {
            length++;
            p = p.next;
        }
        if (length < k) {
            return null;
        }
        int i = 1;
        while (i < length - k + 1) {
            head = head.next;
            i++;
        }
        return head;
    }

    /**
     * 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        // 为空的情况
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        // 结点不空
        ListNode node = null;
        if (list1.val > list2.val) {
            node = list2;
            node.next = Merge(list1, list2.next);
        } else {
            node = list1;
            node.next = Merge(list1.next, list2);
        }
        return node;
    }

    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }

    /**
     * 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
     *
     * @param root1
     * @param root2
     * @return
     */
    public static boolean HasSubtree(TreeNode root1, TreeNode root2) {
        boolean result = false;
        //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
        if (root2 != null && root1 != null) {
            //如果找到了对应Tree2的根节点的点
            if (root1.val == root2.val) {
                //以这个根节点为为起点判断是否包含Tree2
                result = doesTree1HaveTree2(root1, root2);
            }
            //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.left, root2);
            }

            //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
            if (!result) {
                result = HasSubtree(root1.right, root2);
            }
        }
        //返回结果
        return result;
    }

    public static boolean doesTree1HaveTree2(TreeNode node1, TreeNode node2) {
        //如果Tree2已经遍历完了都能对应的上，返回true
        if (node2 == null) {
            return true;
        }
        //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
        if (node1 == null) {
            return false;
        }
        //如果其中有一个点没有对应上，返回false
        if (node1.val != node2.val) {
            return false;
        }

        //如果根节点对应的上，那么就分别去子节点里面匹配
        return doesTree1HaveTree2(node1.left, node2.left) && doesTree1HaveTree2(node1.right, node2.right);
    }

    /**
     * 操作给定的二叉树，将其变换为源二叉树的镜像。
     * 二叉树的镜像定义：源二叉树
     * 8
     * /  \
     * 6   10
     * / \  / \
     * 5  7 9 11
     * 镜像二叉树
     * 8
     * /  \
     * 10   6
     * / \  / \
     * 11 9 7  5
     * 先前序遍历这棵树的每个结点，如果遍历到的结点有子结点，就交换它的两个子节点，
     * 当交换完所有的非叶子结点的左右子结点之后，就得到了树的镜像
     *
     * @param root
     */
    public void Mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            return;
        }

        TreeNode pTemp = root.left;
        root.left = root.right;
        root.right = pTemp;

        if (root.left != null) {
            Mirror(root.left);
        }
        if (root.right != null) {
            Mirror(root.right);
        }

    }


    /**
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
     * 假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
     * 序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
     * 但4,3,5,1,2就不可能是该压栈序列的弹出序列。
     * （注意：这两个序列的长度是相等的）
     *
     * @param pushA
     * @param popA
     * @return
     */
    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length == 0 || popA.length == 0) {
            return false;
        }
        ;

        Stack<Integer> stack = new Stack<>();
        int popIndex = 0;
        for (int i = 0; i < pushA.length; i++) {
            stack.push(pushA[i]);
            //如果栈不为空，且栈顶元素等于弹出序列
            while (!stack.empty() && stack.peek() == popA[popIndex]) {
                //出栈
                stack.pop();
                //弹出序列向后一位
                popIndex++;
            }
        }
        return stack.empty();
    }

    /**
     * 从上往下打印出二叉树的每个节点，同层节点从左至右打印。
     * <p>
     * 用arraylist模拟一个队列来存储相应的TreeNode
     *
     * @param root
     * @return
     */
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        ArrayList<TreeNode> queue = new ArrayList<>();
        if (root == null) {
            return list;
        }
        queue.add(root);
        while (queue.size() != 0) {
            TreeNode temp = queue.remove(0);
            if (temp.left != null) {
                queue.add(temp.left);
            }
            if (temp.right != null) {
                queue.add(temp.right);
            }
            list.add(temp.val);
        }
        return list;
    }

    /**
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
     * （后序遍历：首先遍历左子树，然后遍历右子树，最后访问根结点，
     * 在遍历左、右子树时，仍然先遍历左子树，然后遍历右子树，最后遍历根结点）
     * <p>
     * 解：BST的后序序列的合法序列是，对于一个序列S，最后一个元素是x
     * （也就是根），如果去掉最后一个元素的序列为T，
     * 那么T满足：T可以分成两段，前一段（左子树）小于x，后一段（右子树）大于x，
     * 且这两段（子树）都是合法的后序序列。完美的递归定义 : )
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        int count = sequence.length;
        if (count == 0) {
            return false;
        }
        return isRight(sequence, 0, count - 1);

    }

    public boolean isRight(int[] sequence, int start, int end) {
        if (start >= end) {
            return true;
        }
        int i = end - 1;
        while (sequence[i] > sequence[end] && i > start) {
            i--;
        }
        for (int j = start; j < i; j++) {
            if (sequence[j] > sequence[end]) {
                return false;
            }
        }
        return isRight(sequence, start, i) && isRight(sequence, i + 1, end - 1);


    }

    /**
     * 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
     * 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
     * 由于数字2在数组中出现了5次，超过数组长度的一半，
     * 因此输出2。如果不存在则输出0。
     *
     * @param array
     * @return
     */
    public int MoreThanHalfNum_Solution(int[] array) {
        // 测试用例 [1,2,2]->2; [1,2,3]->0; []->0;  [1]->1; [1,2]->0;
        if (array.length == 0) {
            return 0;
        }
        if (array.length == 1) {
            return array[0];
        }
        // 遍历数组 记录出现的次数
        HashMap<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i < array.length; i++) {
            if (!count.containsKey(array[i])) {
                count.put(array[i], 1);
            } else {
                Integer value = count.get(array[i]);
                count.put(array[i], value + 1);
            }
        }
        // 判断次数是否超过数组长度的一半
        for (Integer i : count.keySet()) {
            if (count.get(i) > array.length / 2) {
                return i;
            }
        }
        return 0;
    }

    /**
     * 快排实现
     *
     * @param num
     * @param left
     * @param right
     * @return
     */

    private static void QuickSort(int[] num, int left, int right) {
        //如果left等于right，即数组只有一个元素，直接返回
        if (left >= right) {
            return;
        }
        //设置最左边的元素为基准值
        int key = num[left];
        //数组中比key小的放在左边，比key大的放在右边，key值下标为i
        int i = left;
        int j = right;
        while (i < j) {
            //j向左移，直到遇到比key小的值
            while (num[j] >= key && i < j) {
                j--;
            }
            //i向右移，直到遇到比key大的值
            while (num[i] <= key && i < j) {
                i++;
            }
            //i和j指向的元素交换
            if (i < j) {
                int temp = num[i];
                num[i] = num[j];
                num[j] = temp;
            }
        }
        num[left] = num[i];
        num[i] = key;
        QuickSort(num, left, i - 1);
        QuickSort(num, i + 1, right);
    }

    /**
     * 时间复杂度O(n)，空间复杂度O(1）
     * 查找数组中重复的数字
     * 在长度为n的数组中，所有的元素都是0到n-1的范围内。
     * 数组中的某些数字是重复的，但不知道有几个重复的数字，
     * 也不知道重复了几次，请找出任意重复的数字。
     * 例如，输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出为2或3
     *
     * @param nums
     * @return
     */
    public static ArrayList<Integer> findDuplicatedNum(int[] nums) {
        ArrayList<Integer> integers = new ArrayList<>();
        if (nums == null || nums.length <= 0) {
            return null;
        }
        // 数据不在0-n-1区间
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] < 0 || nums[i] > nums.length - 1) {
                return null;
            }
        }
        // 数组中的数字为 0 到 n-1 的范围内。如果这个数组中没有重复的数字，则对应的 i 位置的数据也为 i。
        // 可以重排此数组，扫描数组中的每一个数字，当扫描到下标为 i 的数字时，首先比较这个数字（m）是不是等于 i。
        // 如果是，接着扫描下一个数字。如果不是，再拿它和第m 个数字比较，
        // 如果相等则找到重复的数据。否则就把第 i 个数字与第 m 个数字交换。
        // 重复这个比较、交换的过程，直到找到重复的数字。
        for (int i = 0; i < nums.length; ++i) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    integers.add(nums[i]);
                    // 找到该数字后 就结束循环
                    break;
                }
                // 交换数字
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }

        return integers;
    }

    /**
     * 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
     * @param input
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> minKs = new ArrayList<>();
        if(k>input.length){return minKs;}
        QuickSort(input,0,input.length-1);
        for(int i=0;i<k;i++){
            minKs.add(input[i]);
        }
        return minKs;
    }

    public static void main(String args[])
    {

    }





}